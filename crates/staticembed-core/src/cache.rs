//! A bounded, content-addressed cache of embeddings.
//!
//! The key is a SHA-256 over the model's content address and the exact input
//! bytes. Both halves are load bearing:
//!
//! * **Model.** Vectors from two models — or two revisions of one model — are
//!   not comparable, so a cached vector must never survive an asset swap. The
//!   model half of the key is [`crate::model::Model::key`], a digest of the
//!   bundled asset bytes themselves, so the invalidation does not depend on
//!   anyone remembering to bump a version string.
//!
//! * **Text.** The exact UTF-8 bytes, with no case folding, trimming or Unicode
//!   normalisation applied first. A normalisation the encoder does not also
//!   perform would hand back a vector for text that was never embedded, and the
//!   caller would have no way to see it happen.
//!
//! Capacity is bounded because the intended workload is a whole text column. The
//! cache keeps two generations of at most [`EmbeddingCache::capacity`] entries;
//! when the newer one fills, it becomes the older one and a fresh generation
//! starts, so the live set is at most twice the capacity and an entry that keeps
//! being asked for keeps being promoted.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// Domain tag mixed into every cache key.
const CACHE_KEY_DOMAIN: &[u8] = b"staticembed/cache-key/v1";

/// Entries per generation. The live cache holds at most twice this.
///
/// At 256 floats per vector this bounds the cached vectors at roughly 32 MB of
/// payload across both generations. `bounded_cache_never_exceeds_two_generations`
/// pins the entry bound; the byte figure follows from it and the model width.
pub const DEFAULT_CAPACITY: usize = 16_384;

/// A cache key: a 32-byte digest of the model identity and the input text.
pub type CacheKey = [u8; 32];

/// Derive the cache key for `text` under the model identified by `model_key`.
///
/// The text length is written before the text so that no two distinct inputs
/// can be assembled into the same byte stream.
pub fn cache_key(model_key: &[u8; 32], text: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_KEY_DOMAIN);
    hasher.update(model_key);
    hasher.update((text.len() as u64).to_le_bytes());
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

/// What the cache has been doing, as reported to SQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups that had to be embedded.
    pub misses: u64,
    /// Vectors currently held, across both generations.
    pub entries: u64,
    /// Entries per generation.
    pub capacity: u64,
}

/// A two-generation bounded map from cache key to vector.
pub struct EmbeddingCache {
    hot: HashMap<CacheKey, Arc<[f32]>>,
    cold: HashMap<CacheKey, Arc<[f32]>>,
    capacity: usize,
    hits: u64,
    misses: u64,
}

impl EmbeddingCache {
    /// A cache holding at most `capacity` entries per generation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hot: HashMap::new(),
            cold: HashMap::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
        }
    }

    /// Look a key up, counting the outcome.
    ///
    /// A key found in the older generation is promoted, so repeatedly asked-for
    /// vectors survive generation rolls.
    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<[f32]>> {
        if let Some(vector) = self.hot.get(key) {
            self.hits += 1;
            return Some(Arc::clone(vector));
        }
        if let Some(vector) = self.cold.remove(key) {
            self.hits += 1;
            self.insert_hot(*key, Arc::clone(&vector));
            return Some(vector);
        }
        self.misses += 1;
        None
    }

    /// Store a vector against its key.
    pub fn insert(&mut self, key: CacheKey, vector: Arc<[f32]>) {
        self.insert_hot(key, vector);
    }

    fn insert_hot(&mut self, key: CacheKey, vector: Arc<[f32]>) {
        if self.hot.len() >= self.capacity && !self.hot.contains_key(&key) {
            self.cold = std::mem::take(&mut self.hot);
        }
        self.hot.insert(key, vector);
    }

    /// Drop every cached vector, returning how many were dropped.
    ///
    /// Hit and miss counters are reset with the entries: they describe the
    /// contents that just went away.
    pub fn clear(&mut self) -> u64 {
        let dropped = (self.hot.len() + self.cold.len()) as u64;
        self.hot.clear();
        self.cold.clear();
        self.hits = 0;
        self.misses = 0;
        dropped
    }

    /// Current counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: (self.hot.len() + self.cold.len()) as u64,
            capacity: self.capacity as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_key(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn vector(value: f32) -> Arc<[f32]> {
        Arc::from(vec![value; 4].into_boxed_slice())
    }

    /// The same text under two different models keys to two different entries.
    ///
    /// This is AC4's "keyed on content **and** model version". Without it a
    /// model swap would serve vectors from the previous model, which no other
    /// assertion in this crate would notice.
    #[test]
    fn the_model_key_is_part_of_the_cache_key() {
        let under_a = cache_key(&model_key(0xAA), "manufacturing");
        let under_b = cache_key(&model_key(0xBB), "manufacturing");
        assert_ne!(under_a, under_b);
    }

    /// A cached vector is not reachable under a different model key.
    #[test]
    fn a_cached_vector_does_not_survive_a_model_change() {
        let mut cache = EmbeddingCache::with_capacity(8);
        cache.insert(cache_key(&model_key(0xAA), "steel"), vector(1.0));
        assert!(cache.get(&cache_key(&model_key(0xBB), "steel")).is_none());
    }

    #[test]
    fn different_text_under_one_model_keys_differently() {
        let key = model_key(0x11);
        assert_ne!(cache_key(&key, "steel"), cache_key(&key, "steel "));
        assert_ne!(cache_key(&key, "Steel"), cache_key(&key, "steel"));
        assert_ne!(cache_key(&key, ""), cache_key(&key, " "));
    }

    /// Concatenation cannot be confused for a longer string: the length prefix
    /// separates them.
    #[test]
    fn adjacent_inputs_cannot_collide_through_concatenation() {
        let key = model_key(0x22);
        assert_ne!(cache_key(&key, "ab"), cache_key(&key, "a\u{0}b"));
    }

    #[test]
    fn a_hit_returns_the_stored_vector_and_counts_as_a_hit() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x33), "copper");
        cache.insert(key, vector(7.0));
        assert_eq!(cache.get(&key).as_deref(), Some(&[7.0_f32; 4][..]));
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (1, 0, 1));
    }

    #[test]
    fn a_miss_is_counted_and_stores_nothing() {
        let mut cache = EmbeddingCache::with_capacity(8);
        assert!(cache.get(&cache_key(&model_key(0x44), "absent")).is_none());
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses, stats.entries), (0, 1, 0));
    }

    /// The live set stays bounded no matter how many distinct values arrive.
    ///
    /// A column of a million distinct strings must not become a million retained
    /// vectors.
    #[test]
    fn bounded_cache_never_exceeds_two_generations() {
        let capacity = 16;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let key = model_key(0x55);
        for i in 0..capacity * 10 {
            cache.insert(cache_key(&key, &format!("value {i}")), vector(i as f32));
            assert!(
                cache.stats().entries as usize <= capacity * 2,
                "entries {} exceeded {} after {} inserts",
                cache.stats().entries,
                capacity * 2,
                i + 1
            );
        }
    }

    /// A key that keeps being asked for survives a generation roll.
    #[test]
    fn a_repeatedly_requested_key_is_promoted_out_of_the_older_generation() {
        let capacity = 4;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let model = model_key(0x66);
        let hot_key = cache_key(&model, "kept");
        cache.insert(hot_key, vector(9.0));

        // Fill enough to roll the generation twice, touching the key between
        // rolls so it is promoted back into the newer generation each time.
        for round in 0..2 {
            for i in 0..capacity {
                cache.insert(cache_key(&model, &format!("r{round}-{i}")), vector(0.0));
            }
            assert!(
                cache.get(&hot_key).is_some(),
                "the promoted key was evicted after round {round}"
            );
        }
    }

    #[test]
    fn clear_reports_what_it_dropped_and_resets_the_counters() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let model = model_key(0x77);
        cache.insert(cache_key(&model, "one"), vector(1.0));
        cache.insert(cache_key(&model, "two"), vector(2.0));
        let _ = cache.get(&cache_key(&model, "one"));
        assert_eq!(cache.clear(), 2);
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 0,
                misses: 0,
                entries: 0,
                capacity: 8
            }
        );
    }
}
