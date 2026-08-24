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
//!   normalisation applied first. The only normalisation that would be safe is
//!   the one the tokenizer already applies, and reproducing a dependency's
//!   normalisation here would mean a tokenizer bump silently changing which
//!   inputs share an entry. Keying on the exact bytes is correct whatever the
//!   tokenizer does; what it costs is a duplicate entry for inputs that
//!   normalise together, which
//!   `inputs_that_embed_alike_still_occupy_separate_cache_entries` in the crate
//!   root counts.
//!
//! # Why the cache fills and then stops, rather than evicting
//!
//! The workload this exists for is a scan: `SELECT embed(description) FROM t`,
//! then the same query again. Under any recency-ordered eviction — LRU, FIFO,
//! or the two generations this used to keep — a repeated scan of a working set
//! larger than the cache evicts every entry exactly before the scan returns to
//! it, and the hit rate is **zero**, not merely reduced. That is Bélády's
//! cyclic-scan pathology and it is a cliff rather than a slope: measured on the
//! previous build at 16,384 entries per generation, 25,000 distinct values gave
//! 15,538 hits over two passes and 33,000 distinct values gave 0.
//!
//! So this cache admits entries until it is full and then declines new ones.
//! A repeated query over a column with more distinct values than
//! [`EmbeddingCache::capacity`] is served for `capacity` of them and re-embeds
//! the rest — a hit rate of `capacity / distinct` instead of zero, and it does
//! not collapse as the column grows. What it costs is adaptivity: the resident
//! set is whatever was seen first in the session, so a session that moves on to
//! a different column keeps the old one until someone calls
//! `staticembed_cache_clear()`. [`CacheStats::uncached`] is how a caller sees
//! that this is happening rather than inferring it from a slow query.

use std::collections::HashMap;
use std::sync::Arc;

use sha2::{Digest, Sha256};

/// Domain tag mixed into every cache key.
const CACHE_KEY_DOMAIN: &[u8] = b"staticembed/cache-key/v1";

/// Memory the cache may spend, in bytes.
///
/// A budget rather than an entry count, because an entry's size is the model's
/// vector width and a model swap would otherwise change the memory this holds
/// without changing any number written down. 64 MiB is about 61,000 vectors at
/// this model's width; [`capacity_for_budget`] is where that conversion lives
/// and `the_default_budget_holds_at_least_fifty_thousand_vectors` is what stops
/// the figure drifting.
pub const DEFAULT_BUDGET_BYTES: usize = 64 * 1024 * 1024;

/// A cache key: a 32-byte digest of the model identity and the input text.
pub type CacheKey = [u8; 32];

/// Bytes one cached vector costs, at a given model width.
///
/// The key and the `Arc` pointer sit inline in the map's bucket array, which
/// `HashMap` keeps under seven-eighths full and pairs with one control byte per
/// bucket; the floats and the `Arc`'s two reference counts sit on the heap.
/// This is an estimate of a real allocator's behaviour, not an exact figure —
/// it is used to pick a capacity, so being close is what it needs to be.
pub fn entry_bytes(dim: usize) -> usize {
    const CONTROL_BYTE: usize = 1;
    const ARC_COUNTS: usize = 2 * std::mem::size_of::<usize>();
    let inline = std::mem::size_of::<CacheKey>() + std::mem::size_of::<Arc<[f32]>>() + CONTROL_BYTE;
    // Seven-eighths load factor: eight buckets of storage buy seven entries.
    let inline_with_slack = inline.div_ceil(7) * 8;
    inline_with_slack + ARC_COUNTS + dim * std::mem::size_of::<f32>()
}

/// How many vectors of width `dim` fit in `budget_bytes`, never fewer than one.
pub fn capacity_for_budget(budget_bytes: usize, dim: usize) -> usize {
    (budget_bytes / entry_bytes(dim)).max(1)
}

/// What the cache has been doing, as reported to SQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups answered from the cache.
    pub hits: u64,
    /// Lookups that had to be embedded.
    pub misses: u64,
    /// Lookups the cache declined to store because it was full.
    ///
    /// Non-zero means the column has more distinct values than the cache can
    /// hold, so a repeated query will re-embed the excess.
    pub uncached: u64,
    /// Vectors currently held.
    pub entries: u64,
    /// The most vectors this cache will hold.
    pub capacity: u64,
}

/// A bounded map from cache key to vector that fills and then stops.
pub struct EmbeddingCache {
    entries: HashMap<CacheKey, Arc<[f32]>>,
    capacity: usize,
    hits: u64,
    misses: u64,
    uncached: u64,
}

impl EmbeddingCache {
    /// A cache holding at most `capacity` vectors.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
            uncached: 0,
        }
    }

    /// Look a key up, counting the outcome.
    pub fn get(&mut self, key: &CacheKey) -> Option<Arc<[f32]>> {
        match self.entries.get(key) {
            Some(vector) => {
                self.hits += 1;
                Some(Arc::clone(vector))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Look again after another thread may have filled the key in.
    ///
    /// This is the second look one caller takes for one row, so it does not
    /// count as a fresh lookup. If the value is there now, the miss that caller
    /// already recorded was stale and becomes a hit — which is what keeps
    /// `hits + misses` equal to the number of rows, and `hits` equal to the
    /// number of rows the cache actually answered.
    pub fn recheck(&mut self, key: &CacheKey) -> Option<Arc<[f32]>> {
        let vector = self.entries.get(key).map(Arc::clone)?;
        self.misses = self.misses.saturating_sub(1);
        self.hits += 1;
        Some(vector)
    }

    /// True when the cache will not take any further distinct key.
    pub fn is_full(&self) -> bool {
        self.entries.len() >= self.capacity
    }

    /// Store a vector against its key, if there is room.
    ///
    /// Returns whether it was stored. A full cache declines rather than
    /// evicting; see the module docs for why.
    ///
    /// The full-cache branch here is an invariant guard on a public method, not
    /// the production path: [`crate::embed`] tests [`Self::is_full`] itself and
    /// never reaches this when the answer is yes, because it also has to decide
    /// whether to take a flight. Mutating this branch leaves the SQL suite
    /// green, which is how that was established.
    pub fn insert(&mut self, key: CacheKey, vector: Arc<[f32]>) -> bool {
        if self.is_full() && !self.entries.contains_key(&key) {
            self.uncached += 1;
            return false;
        }
        self.entries.insert(key, vector);
        true
    }

    /// Record that a lookup had to be embedded because the cache was full.
    ///
    /// The caller short-circuits [`Self::insert`] when [`Self::is_full`] is
    /// already true, so the count has to be made here instead.
    pub fn note_uncached(&mut self) {
        self.uncached += 1;
    }

    /// Drop every cached vector, returning how many were dropped.
    ///
    /// The counters go with the entries: they describe the contents that just
    /// went away.
    pub fn clear(&mut self) -> u64 {
        let dropped = self.entries.len() as u64;
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.uncached = 0;
        dropped
    }

    /// Current counters.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            uncached: self.uncached,
            entries: self.entries.len() as u64,
            capacity: self.capacity as u64,
        }
    }
}

/// Derive the cache key for `text` under the model identified by `model_key`.
///
/// `text` is written last and every field before it is fixed width, so two
/// different inputs cannot assemble into the same byte stream. A field added
/// after `text` would break that and would need a length prefix in front of it.
pub fn cache_key(model_key: &[u8; 32], text: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(CACHE_KEY_DOMAIN);
    hasher.update(model_key);
    hasher.update(text.as_bytes());
    hasher.finalize().into()
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

    /// The text half of the key is the exact input bytes.
    ///
    /// Every pair below differs only by case, whitespace or Unicode
    /// composition, and this model's tokenizer folds all three away — so these
    /// are pairs the encoder gives the SAME vector for, and the cache still
    /// gives them two entries. That is the deliberate direction to be wrong in:
    /// a duplicate entry costs memory, while a key that collapsed two inputs
    /// the tokenizer did not would return a vector nobody asked for.
    #[test]
    fn the_cache_key_uses_the_exact_input_bytes() {
        let key = model_key(0x11);
        assert_ne!(cache_key(&key, "steel"), cache_key(&key, "steel "));
        assert_ne!(cache_key(&key, "Steel"), cache_key(&key, "steel"));
        assert_ne!(cache_key(&key, ""), cache_key(&key, " "));
        assert_ne!(cache_key(&key, "cafe\u{301}"), cache_key(&key, "caf\u{e9}"));
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

    /// A successful recheck turns the caller's stale miss into a hit, and adds
    /// no second lookup.
    #[test]
    fn a_successful_recheck_corrects_the_stale_miss_it_follows() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x55), "nickel");

        assert!(cache.get(&key).is_none());
        assert_eq!((cache.stats().hits, cache.stats().misses), (0, 1));

        // Another thread fills it in while this caller is waiting.
        cache.insert(key, vector(1.0));
        assert!(cache.recheck(&key).is_some());
        assert_eq!(
            (cache.stats().hits, cache.stats().misses),
            (1, 0),
            "one row, one lookup, and the cache answered it"
        );
    }

    /// A failed recheck moves nothing: the miss stands and the caller encodes.
    #[test]
    fn a_failed_recheck_leaves_the_counters_alone() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let key = cache_key(&model_key(0x56), "absent");
        assert!(cache.get(&key).is_none());
        assert!(cache.recheck(&key).is_none());
        assert_eq!((cache.stats().hits, cache.stats().misses), (0, 1));
    }

    /// The live set stays bounded no matter how many distinct values arrive.
    #[test]
    fn the_cache_never_holds_more_than_its_capacity() {
        let capacity = 16;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let key = model_key(0x66);
        for i in 0..capacity * 10 {
            cache.insert(cache_key(&key, &format!("value {i}")), vector(i as f32));
            assert!(
                cache.stats().entries as usize <= capacity,
                "entries {} exceeded {capacity} after {} inserts",
                cache.stats().entries,
                i + 1
            );
        }
        assert_eq!(cache.stats().entries as usize, capacity);
        assert_eq!(cache.stats().uncached, (capacity * 9) as u64);
    }

    /// **A repeated scan of a working set larger than the cache does not fall
    /// to a zero hit rate.**
    ///
    /// This is the assertion the previous design had none of. Two sequential
    /// passes over 200 distinct keys through a 64-entry cache: the second pass
    /// must hit for the 64 that are resident. Under the two-generation policy
    /// this replaced, and under LRU or FIFO, it hits for none of them, because
    /// a cyclic scan evicts every entry exactly before it is next needed.
    #[test]
    fn a_repeated_scan_larger_than_the_cache_still_hits_for_a_full_cache_worth() {
        let capacity = 64;
        let distinct = 200;
        let mut cache = EmbeddingCache::with_capacity(capacity);
        let model = model_key(0x77);
        let keys: Vec<CacheKey> = (0..distinct)
            .map(|i| cache_key(&model, &format!("value {i}")))
            .collect();

        // First pass: every key is looked up and stored if there is room.
        for (i, key) in keys.iter().enumerate() {
            if cache.get(key).is_none() {
                cache.insert(*key, vector(i as f32));
            }
        }
        let after_first = cache.stats();
        assert_eq!(after_first.entries as usize, capacity);

        // Second pass, in the same order — the pattern a table rescan produces.
        for key in &keys {
            let _ = cache.get(key);
        }
        let hits_in_second_pass = cache.stats().hits - after_first.hits;
        assert_eq!(
            hits_in_second_pass as usize, capacity,
            "a rescan must be served for a full cache worth, not for none of it"
        );
    }

    /// A value the cache declined is still declined the next time round, and is
    /// counted every time.
    #[test]
    fn a_full_cache_declines_and_says_how_often() {
        let mut cache = EmbeddingCache::with_capacity(2);
        let model = model_key(0x88);
        assert!(cache.insert(cache_key(&model, "one"), vector(1.0)));
        assert!(cache.insert(cache_key(&model, "two"), vector(2.0)));
        assert!(!cache.insert(cache_key(&model, "three"), vector(3.0)));
        assert!(!cache.insert(cache_key(&model, "three"), vector(3.0)));
        assert_eq!(cache.stats().uncached, 2);
        // A key already held is still updatable — a full cache is closed to new
        // keys, not to the ones it has.
        assert!(cache.insert(cache_key(&model, "one"), vector(9.0)));
        assert_eq!(
            cache.get(&cache_key(&model, "one")).as_deref(),
            Some(&[9.0_f32; 4][..])
        );
    }

    #[test]
    fn clear_reports_what_it_dropped_and_resets_the_counters() {
        let mut cache = EmbeddingCache::with_capacity(8);
        let model = model_key(0x99);
        cache.insert(cache_key(&model, "one"), vector(1.0));
        cache.insert(cache_key(&model, "two"), vector(2.0));
        let _ = cache.get(&cache_key(&model, "one"));
        assert_eq!(cache.clear(), 2);
        assert_eq!(
            cache.stats(),
            CacheStats {
                hits: 0,
                misses: 0,
                uncached: 0,
                entries: 0,
                capacity: 8
            }
        );
    }

    /// The capacity is the largest number of entries that fits the budget.
    ///
    /// Stated as the exact property rather than as a ratio, so it holds at
    /// every width instead of at the one that was convenient to check.
    #[test]
    fn the_capacity_is_the_largest_entry_count_that_fits_the_budget() {
        for dim in [4, 256, 1024] {
            for budget in [64 * 1024, 1024 * 1024, 64 * 1024 * 1024] {
                let capacity = capacity_for_budget(budget, dim);
                let cost = entry_bytes(dim);
                assert!(capacity * cost <= budget, "dim {dim} budget {budget}");
                assert!((capacity + 1) * cost > budget, "dim {dim} budget {budget}");
            }
        }
    }

    /// The budget converts to a capacity that scales with the model width.
    #[test]
    fn a_wider_model_gets_fewer_entries_from_the_same_budget() {
        let budget = 64 * 1024 * 1024;
        let narrow = capacity_for_budget(budget, 256);
        let wide = capacity_for_budget(budget, 1024);
        assert!(narrow > wide, "{narrow} should exceed {wide}");
    }

    #[test]
    fn halving_the_budget_roughly_halves_the_capacity() {
        let full = capacity_for_budget(64 * 1024 * 1024, 256);
        let half = capacity_for_budget(32 * 1024 * 1024, 256);
        assert_eq!(full / 2, half);
    }

    #[test]
    fn a_budget_too_small_for_one_vector_still_gives_a_usable_cache() {
        assert_eq!(capacity_for_budget(1, 256), 1);
        assert_eq!(capacity_for_budget(0, 256), 1);
    }

    /// **The default budget is big enough to be worth having.**
    ///
    /// A capacity constant nothing pins is free to be anything: setting the old
    /// `DEFAULT_CAPACITY` to 4 left the entire suite green. This fixes the floor
    /// in the units a user cares about — how many distinct values of a column
    /// can be held — and `crate::tests::a_repeated_query_over_fifty_thousand
    /// _distinct_values_re_embeds_none_of_them` spends it.
    #[test]
    fn the_default_budget_holds_at_least_fifty_thousand_vectors() {
        let capacity = capacity_for_budget(DEFAULT_BUDGET_BYTES, 256);
        assert!(
            capacity >= 50_000,
            "the default budget holds only {capacity} vectors of width 256"
        );
    }
}
