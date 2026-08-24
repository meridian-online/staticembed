//! Static text embeddings from a model bundled at build time.
//!
//! This crate is the engine: it owns the embedded model, the embedding call and
//! the cache. It knows nothing about DuckDB, which is what lets every behaviour
//! below be tested with plain `cargo test`.
//!
//! # The contract, in one place
//!
//! * [`embed`] returns a vector of [`dim`] `f32` values for any `&str`.
//! * Text with no in-vocabulary tokens — the empty string, whitespace, a string
//!   of symbols the vocabulary does not carry — returns a **zero vector of full
//!   width**, not an absent value and not an error. SQL `NULL` is not text and
//!   never reaches this crate; the DuckDB layer propagates it.
//! * Repeating a call for the same text under the same model does not re-embed.
//!   [`Stats::encoded`] is the count of actual encoder invocations and is the
//!   observable that says so.
//! * Nothing here reads a file or opens a socket after compilation.

pub mod cache;
pub mod model;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use cache::{CacheStats, EmbeddingCache};

/// Extension version, shared by every crate in the workspace.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Number of times the encoder has actually run since the last [`clear_cache`].
static ENCODED: AtomicU64 = AtomicU64::new(0);

static CACHE: OnceLock<Mutex<EmbeddingCache>> = OnceLock::new();

fn cache() -> MutexGuard<'static, EmbeddingCache> {
    let cache =
        CACHE.get_or_init(|| Mutex::new(EmbeddingCache::with_capacity(cache::DEFAULT_CAPACITY)));
    // A panic inside the extension would otherwise poison the cache for the rest
    // of the session; the data behind the lock is a cache, so recovering it is
    // always safe.
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Everything SQL can observe about the cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Lookups answered without embedding.
    pub hits: u64,
    /// Lookups that were not in the cache.
    pub misses: u64,
    /// Times the encoder actually ran.
    pub encoded: u64,
    /// Vectors currently held.
    pub entries: u64,
    /// Entries per cache generation.
    pub capacity: u64,
}

/// Width of every vector this build returns.
pub fn dim() -> Result<usize, &'static str> {
    Ok(model::bundled()?.dim())
}

/// Embed one string, reusing an earlier result when the same text has been seen
/// under the same model.
///
/// See the crate docs for what comes back for text with no tokens.
pub fn embed(text: &str) -> Result<Arc<[f32]>, &'static str> {
    let model = model::bundled()?;
    let key = cache::cache_key(model.key(), text);

    if let Some(vector) = cache().get(&key) {
        return Ok(vector);
    }

    // The encoder runs outside the lock: it is the expensive part and it is pure,
    // so two threads racing on the same text compute the same vector and the
    // second insert is a no-op. `ENCODED` counts real encoder invocations, which
    // is the number AC4 is about.
    ENCODED.fetch_add(1, Ordering::Relaxed);
    let vector: Arc<[f32]> = Arc::from(model.embed(text).into_boxed_slice());
    cache().insert(key, Arc::clone(&vector));
    Ok(vector)
}

/// Embed one string without consulting or filling the cache.
pub fn embed_uncached(text: &str) -> Result<Vec<f32>, &'static str> {
    ENCODED.fetch_add(1, Ordering::Relaxed);
    Ok(model::bundled()?.embed(text))
}

/// Current cache counters.
pub fn stats() -> Stats {
    let CacheStats {
        hits,
        misses,
        entries,
        capacity,
    } = cache().stats();
    Stats {
        hits,
        misses,
        encoded: ENCODED.load(Ordering::Relaxed),
        entries,
        capacity,
    }
}

/// Drop every cached vector and reset the counters, returning the number of
/// vectors dropped.
pub fn clear_cache() -> u64 {
    ENCODED.store(0, Ordering::Relaxed);
    cache().clear()
}

/// One line naming this build: the extension version, the bundled model, and the
/// width of the vectors it returns.
pub fn describe() -> String {
    match model::bundled() {
        Ok(model) => format!(
            "staticembed {} (model {}@{}, key {}, dim {})",
            VERSION,
            model::MODEL_ID,
            &model::MODEL_REVISION[..12],
            &model.key_hex()[..12],
            model.dim()
        ),
        Err(message) => format!("staticembed {VERSION} (model unavailable: {message})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache and the counters are process-global, so the tests that read
    /// them run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// AC4, at the engine: asking for the same text again does not re-embed.
    ///
    /// `encoded` counts encoder invocations, not cache bookkeeping, so this
    /// fails if the cache is consulted but not honoured.
    #[test]
    fn repeating_a_value_does_not_re_embed_it() {
        let _guard = serial();
        clear_cache();

        let column = ["manufacturing", "logistics", "manufacturing", "logistics"];
        for value in column {
            embed(value).expect("embed");
        }
        let first_pass = stats();
        assert_eq!(first_pass.encoded, 2, "two distinct values, two encodes");

        for value in column {
            embed(value).expect("embed");
        }
        let second_pass = stats();
        assert_eq!(
            second_pass.encoded, first_pass.encoded,
            "a repeated query must not re-embed"
        );
        assert!(second_pass.hits > first_pass.hits);
    }

    /// A cache hit returns the same vector the encoder produced, not a stale or
    /// empty one.
    #[test]
    fn a_cached_vector_equals_the_uncached_one() {
        let _guard = serial();
        clear_cache();
        let text = "a supplier of precision bearings";
        let cached = embed(text).expect("embed");
        let direct = embed_uncached(text).expect("embed");
        assert_eq!(cached.as_ref(), direct.as_slice());
    }

    #[test]
    fn clearing_the_cache_drops_the_entries_and_the_counters() {
        let _guard = serial();
        clear_cache();
        embed("one").expect("embed");
        embed("two").expect("embed");
        assert_eq!(clear_cache(), 2);
        assert_eq!(
            stats(),
            Stats {
                hits: 0,
                misses: 0,
                encoded: 0,
                entries: 0,
                capacity: cache::DEFAULT_CAPACITY as u64,
            }
        );
    }

    /// The tokenizer folds case, surrounding whitespace and Unicode composition
    /// away, so these inputs are one value to the model.
    ///
    /// Pinned because it is a property of the bundled tokenizer rather than of
    /// this code: a caller may rely on `embed(name)` matching `embed(NAME)`, and
    /// a tokenizer swap that changed it would change the SQL contract with
    /// nothing else reddening.
    #[test]
    fn case_and_surrounding_whitespace_do_not_change_the_vector() {
        let _guard = serial();
        let pairs = [
            ("steel", "STEEL"),
            ("steel", "  steel  "),
            ("caf\u{e9}", "cafe\u{301}"),
        ];
        for (left, right) in pairs {
            assert_eq!(
                embed(left).expect("embed").as_ref(),
                embed(right).expect("embed").as_ref(),
                "{left:?} and {right:?} should embed alike",
            );
        }
    }

    /// Word order does not change the vector, and repetition does.
    ///
    /// A Model2Vec vector is the mean of its token vectors, so a phrase and its
    /// shuffle land in the same place. That is the model, not a defect, and it
    /// is pinned here because it bounds what the vectors can be asked to do.
    #[test]
    fn the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not() {
        let _guard = serial();
        assert_eq!(
            embed("valve bodies").expect("embed").as_ref(),
            embed("bodies valve").expect("embed").as_ref(),
        );
        assert_ne!(
            embed("steel steel copper").expect("embed").as_ref(),
            embed("steel copper").expect("embed").as_ref(),
        );
    }

    /// Those pairs are nevertheless separate cache entries, because the key is
    /// the exact bytes. This is the cost of not reproducing the tokenizer's
    /// normalisation in the cache layer, counted rather than left implicit.
    #[test]
    fn inputs_that_embed_alike_still_occupy_separate_cache_entries() {
        let _guard = serial();
        clear_cache();
        embed("steel").expect("embed");
        embed("STEEL").expect("embed");
        assert_eq!(stats().entries, 2);
        assert_eq!(stats().encoded, 2);
    }

    #[test]
    fn every_vector_has_the_declared_width() {
        let _guard = serial();
        let width = dim().expect("dim");
        for text in ["", "one", "a longer sentence about steel stockholders"] {
            assert_eq!(
                embed(text).expect("embed").len(),
                width,
                "width for {text:?}"
            );
        }
    }

    #[test]
    fn describe_names_the_bundled_model_and_the_width() {
        let text = describe();
        assert!(text.contains(model::MODEL_ID), "{text}");
        assert!(text.contains(VERSION), "{text}");
        assert!(
            text.contains(&format!("dim {}", dim().expect("dim"))),
            "{text}"
        );
    }
}
