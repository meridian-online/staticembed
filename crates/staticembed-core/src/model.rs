//! The bundled model, and the one place a string becomes a vector.
//!
//! The three asset files are compiled into the binary with `include_bytes!` and
//! handed to `model2vec_rs` as bytes. Nothing here opens a file at runtime and
//! nothing here opens a socket: `StaticModel::from_bytes` is the only loader
//! this crate calls, and the `local-only` feature on `model2vec-rs` compiles the
//! download path out of the dependency entirely.

use std::sync::OnceLock;

use model2vec_rs::model::StaticModel;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

/// The Hugging Face repository the bundled assets came from.
pub const MODEL_ID: &str = "minishlab/potion-base-8M";

/// The exact revision of that repository the bundled assets were taken at.
pub const MODEL_REVISION: &str = "bf8b056651a2c21b8d2565580b8569da283cab23";

/// SHA-256 of `models/potion-base-8M/model.safetensors` as published.
pub const WEIGHTS_SHA256: &str = "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2";
/// SHA-256 of `models/potion-base-8M/tokenizer.json` as published.
pub const TOKENIZER_SHA256: &str =
    "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50";
/// SHA-256 of `models/potion-base-8M/config.json` as published.
pub const CONFIG_SHA256: &str = "2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553";

const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/model.safetensors");
const TOKENIZER: &[u8] = include_bytes!("../../../models/potion-base-8M/tokenizer.json");
const CONFIG: &[u8] = include_bytes!("../../../models/potion-base-8M/config.json");

/// The token cap [`Model::embed`] truncates to before pooling: text that
/// tokenises to more in-vocabulary ids than this has the excess dropped
/// before the mean, and — because the cut this crate performs also mirrors
/// the character-level pre-cut `model2vec_rs::StaticModel::encode` applies
/// ahead of tokenising — text made of long, sparse tokens can lose content
/// well before it reaches 512 of them. See [`Truncation`] for the mechanism
/// and [`Model::is_truncated`] for how a caller finds out.
///
/// This is `model2vec_rs::StaticModel::encode`'s own default, `Some(512)`,
/// declared here so this crate has exactly one place that says "512".
pub const MAX_TOKENS: usize = 512;

/// Domain tag mixed into the model key so the digest cannot be confused with a
/// plain SHA-256 of any one asset.
const MODEL_KEY_DOMAIN: &[u8] = b"staticembed/model-key/v1";

/// Derive a model's content address from the three asset files it is made of.
///
/// Taken as arguments rather than read from the embedded constants so that a
/// test can hand it altered bytes and see the key move. A version of this that
/// hashed only some of its arguments would pass any test that recomputed the
/// same fields beside it; it cannot pass one that calls it twice.
pub fn model_key(tokenizer: &[u8], weights: &[u8], config: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(MODEL_KEY_DOMAIN);
    hasher.update(MODEL_ID.as_bytes());
    hasher.update([0u8]);
    hasher.update(MODEL_REVISION.as_bytes());
    hasher.update([0u8]);
    hasher.update(tokenizer);
    hasher.update(weights);
    hasher.update(config);
    hasher.finalize().into()
}

/// A loaded static embedding model.
pub struct Model {
    inner: StaticModel,
    dim: usize,
    key: [u8; 32],
    truncation: Truncation,
}

/// The one place this crate decides how much of a text `embed` will use.
///
/// `model2vec_rs::StaticModel::encode_with_args` performs two cuts when asked
/// for at most `max_tokens` ids: first a **character**-level pre-cut of the
/// raw string, to `max_tokens * median_token_length` characters — a
/// performance guard, so an arbitrarily long text is not tokenised in full
/// just to have the result thrown away — and only then a **token**-level cut
/// of the tokenised, out-of-vocabulary-filtered result, to `max_tokens` ids.
/// Both steps are private with no accessor, and an earlier version of this
/// type reimplemented them separately from what decided `is_truncated`: two
/// copies of the same arithmetic, expected to stay in step, with nothing
/// forcing them to. One of those copies missed the character cut entirely,
/// and a fix that recomputed it independently coincided with the token cap on
/// a text built of one token repeated, reporting "not clipped" no matter how
/// long the text ran.
///
/// So this type does not answer a question about `embed`; it **performs the
/// truncation `embed` uses**. [`Model::embed`] calls [`Truncation::surviving_prefix`]
/// to get the exact text it will tokenise, and re-tokenises that with no
/// further limit — the character and token cuts both already applied.
/// [`Model::is_truncated`] asks the same method the same question and compares
/// lengths. There is one implementation of "how much of this text survives",
/// and both callers read it; neither can drift from what the other sees,
/// because they are the same call.
///
/// It carries its own [`Tokenizer`], loaded from the identical `TOKENIZER`
/// bytes [`Model`] itself loads from, because `model2vec_rs::StaticModel`
/// keeps its tokenizer private with no accessor.
struct Truncation {
    tokenizer: Tokenizer,
    /// The id `model2vec_rs` drops from a token list before truncating and
    /// pooling it. Recomputed here the same way its own (private) metadata
    /// step does: there is no accessor for it either.
    unk_token_id: Option<usize>,
    /// The median byte length of a vocabulary token, used to reproduce
    /// `model2vec_rs`'s private character-level pre-cut. Computed the same
    /// way its own (private) `compute_metadata` does: byte lengths of every
    /// vocabulary token, sorted, middle element.
    median_token_length: usize,
}

impl Truncation {
    fn from_tokenizer_bytes(bytes: &[u8]) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("the truncation tokenizer did not load: {e}"))?;

        // No accessor exists for the unk token either, so recover it the way
        // `model2vec_rs`'s private `compute_metadata` does: round-trip the
        // tokenizer through JSON and read `model.unk_token`.
        let spec: serde_json::Value = serde_json::to_value(&tokenizer)
            .map_err(|e| format!("could not inspect the tokenizer for its unk token: {e}"))?;
        let unk_token_id = spec
            .get("model")
            .and_then(|model| model.get("unk_token"))
            .and_then(serde_json::Value::as_str)
            .and_then(|token| tokenizer.token_to_id(token))
            .map(|id| id as usize);

        let mut lens: Vec<usize> = tokenizer.get_vocab(false).keys().map(String::len).collect();
        lens.sort_unstable();
        let median_token_length = lens.get(lens.len() / 2).copied().unwrap_or(1);

        Ok(Self {
            tokenizer,
            unk_token_id,
            median_token_length,
        })
    }

    /// Reproduce `model2vec_rs`'s private character-level pre-cut: at most
    /// `max_tokens * median_token_length` characters of `text`, on a char
    /// boundary.
    fn char_truncate(text: &str, max_tokens: usize, median_token_length: usize) -> &str {
        text.char_indices()
            .nth(max_tokens.saturating_mul(median_token_length))
            .map_or(text, |(byte_idx, _)| &text[..byte_idx])
    }

    /// The exact prefix of `text` that survives both of `model2vec_rs`'s
    /// truncation steps — the substring [`Model::embed`] actually tokenises.
    ///
    /// Applies the character cut, tokenises what remains, drops
    /// out-of-vocabulary ids the way `encode_with_args` does, and — if more
    /// than `max_tokens` ids are still left — cuts back to the byte offset
    /// where the `max_tokens`-th surviving id ends. `Encoding::get_offsets`
    /// reports that offset in the *original* (pre-normalisation) string, which
    /// is what makes slicing `text` at it, rather than the encoder's internal
    /// token buffer, exact.
    ///
    /// This calls [`Tokenizer::encode`], not `encode_fast`: `encode_fast`
    /// documents that it "does not compute offsets", and returns `(0, 0)` for
    /// every one — a first version of this method used it and sliced every
    /// over-cap text down to an empty string, which `embed` then reported as
    /// the zero vector. `model2vec_rs`'s own pooling path uses the fast,
    /// offset-free encode (it only ever needs ids), so this is the one place
    /// in this crate that pays for the slower call, and only because this is
    /// the one place that needs where a token sits in the original string.
    fn surviving_prefix<'a>(&self, text: &'a str, max_tokens: usize) -> &'a str {
        let char_cut = Self::char_truncate(text, max_tokens, self.median_token_length);
        let Ok(encoding) = self.tokenizer.encode(char_cut, false) else {
            // A failure here fails inside `embed` too, when it re-tokenises
            // this same text — this degrades rather than panics because it
            // also backs the `is_truncated` predicate, which has no fallible
            // path to report through.
            return char_cut;
        };

        let ids = encoding.get_ids();
        let offsets = encoding.get_offsets();
        let surviving: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|&(_, &id)| self.unk_token_id.is_none_or(|unk| id as usize != unk))
            .map(|(i, _)| i)
            // One more than the cap is all that's needed to know the cap was
            // reached; collecting further would count ids the pooled mean
            // never sees anyway.
            .take(max_tokens + 1)
            .collect();

        if surviving.len() <= max_tokens {
            // `token_ids.truncate(max_tok)` is a no-op here: nothing past the
            // last kept id is dropped, so the char cut was the only cut, if
            // any. Trailing text that produced no surviving id — whitespace, a
            // trailing run of out-of-vocabulary symbols — is not content that
            // was dropped: it would never have reached the mean regardless.
            return char_cut;
        }

        // More than `max_tokens` ids survived the filter, so `encode_with_args`
        // truncates the list to the first `max_tokens` of them before pooling.
        // Cut the text back to where that last kept id ends.
        &char_cut[..offsets[surviving[max_tokens - 1]].1]
    }

    /// True if [`Model::embed`] discards part of `text`: the prefix that
    /// survives both truncation steps is shorter than `text` itself.
    ///
    /// A `false` for text with no in-vocabulary tokens at all — the empty
    /// string, whitespace, symbols outside the vocabulary — is correct, not a
    /// false negative: nothing was discarded, there was simply nothing there.
    fn truncates(&self, text: &str, max_tokens: usize) -> bool {
        self.surviving_prefix(text, max_tokens).len() < text.len()
    }
}

static BUNDLED: OnceLock<Result<Model, String>> = OnceLock::new();

/// The model compiled into this binary, loaded on first call and reused after.
///
/// Loading is a parse of the embedded bytes — no filesystem lookup, no network,
/// no environment variable, and no configuration a caller has to supply.
pub fn bundled() -> Result<&'static Model, &'static str> {
    match BUNDLED.get_or_init(Model::from_embedded_bytes) {
        Ok(model) => Ok(model),
        Err(message) => Err(message.as_str()),
    }
}

impl Model {
    /// Load the model from the bytes embedded in this binary.
    fn from_embedded_bytes() -> Result<Self, String> {
        let inner = StaticModel::from_bytes(TOKENIZER, WEIGHTS, CONFIG, None)
            .map_err(|e| format!("the bundled {MODEL_ID} assets did not load: {e}"))?;

        // The dimension is read from the model rather than declared here, so a
        // future asset swap cannot leave a stale constant behind.
        let dim = inner.encode_single("dimension probe").len();
        if dim == 0 {
            return Err(format!(
                "the bundled {MODEL_ID} assets produced a zero-width embedding"
            ));
        }

        let key = model_key(TOKENIZER, WEIGHTS, CONFIG);
        let truncation = Truncation::from_tokenizer_bytes(TOKENIZER)?;

        Ok(Model {
            inner,
            dim,
            key,
            truncation,
        })
    }

    /// Number of floats in every vector this model returns.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The model's content address: a SHA-256 over its identity, its pinned
    /// revision and the three bundled asset files.
    ///
    /// This is the "model version" half of the cache key. It is derived from the
    /// bytes rather than from a version string, so replacing the weights
    /// invalidates every cached vector even if nobody remembers to bump a
    /// number.
    pub fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// The model key as lowercase hex.
    pub fn key_hex(&self) -> String {
        hex(&self.key)
    }

    /// Embed one string.
    ///
    /// # What comes back for text with no tokens
    ///
    /// A zero vector of [`Model::dim`] floats — not `None`, not an error, and
    /// not a short vector. That is the case for the empty string, for
    /// whitespace, and for text made only of tokens the vocabulary does not
    /// contain: the tokenizer yields no in-vocabulary ids, so the mean over
    /// zero ids is the zero vector.
    ///
    /// This function does not decide what a SQL `NULL` means. `NULL` never
    /// reaches here — the DuckDB layer propagates it, so `embed(NULL)` is
    /// `NULL`. A caller that wants the zero vector for a missing value asks for
    /// it: `embed(coalesce(t, ''))`.
    ///
    /// A vector of any width other than [`Model::dim`] or zero is a bug in the
    /// encoder, and comes back as an error rather than as silent zeros: a query
    /// that fails is recoverable, and a column of zero vectors that looks like
    /// data is not.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        // `Truncation::surviving_prefix` has already applied both of
        // `model2vec_rs`'s cuts (character, then token), so `encode_with_args`
        // is called with `None` here rather than `Some(MAX_TOKENS)` — a second
        // limit would either be redundant or, worse, a second place this crate
        // could disagree with itself about how much of `text` it used.
        let used = self.truncation.surviving_prefix(text, MAX_TOKENS);
        let sentence = [used.to_string()];
        let vector = self
            .inner
            .encode_with_args(&sentence, None, 1)
            .into_iter()
            .next()
            .unwrap_or_default();
        conform(vector, self.dim)
    }

    /// Whether embedding `text` discards content: the prefix [`Model::embed`]
    /// actually tokenises is shorter than `text`, so the excess never reached
    /// the mean.
    ///
    /// The vector `embed` returns is full width and unit norm whether or not
    /// this is true — nothing about it says content was dropped, which is the
    /// whole reason to ask first.
    pub fn is_truncated(&self, text: &str) -> bool {
        self.truncation.truncates(text, MAX_TOKENS)
    }
}

/// Apply the width contract to whatever the encoder returned.
///
/// A pure function rather than a `match` inline in [`Model::embed`], because the
/// error arm is unreachable from outside the crate and a test that could only
/// call `embed` could never reach it.
fn conform(vector: Vec<f32>, dim: usize) -> Result<Vec<f32>, String> {
    match vector.len() {
        width if width == dim => Ok(vector),
        // The no-token contract: nothing to average is the zero vector, at the
        // model's full width.
        0 => Ok(vec![0.0_f32; dim]),
        width => Err(format!(
            "the encoder returned {width} floats for a model {dim} wide"
        )),
    }
}

/// Lowercase hex of a byte string.
pub fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    /// The embedded assets are the published release, byte for byte.
    ///
    /// AC5's quality figures were measured against `potion-base-8M`. A swapped,
    /// truncated or re-serialised asset would keep every other test green while
    /// invalidating every published claim, so the bytes are pinned here.
    #[test]
    fn bundled_asset_digests_match_the_pinned_release() {
        assert_eq!(digest(WEIGHTS), WEIGHTS_SHA256, "model.safetensors");
        assert_eq!(digest(TOKENIZER), TOKENIZER_SHA256, "tokenizer.json");
        assert_eq!(digest(CONFIG), CONFIG_SHA256, "config.json");
    }

    /// The width the encoder produces is the width the model's own config
    /// declares.
    ///
    /// Everything else that mentions the width takes it from `Model::dim`,
    /// which is itself measured by encoding a probe string — so an encoder that
    /// returned five floats for everything would set `dim` to five and satisfy
    /// every one of those assertions. `config.json` is a second, independent
    /// statement of the same fact, published with the weights, and this is the
    /// only place the two are made to agree.
    #[test]
    fn the_encoder_width_matches_the_width_the_config_declares() {
        let text = std::str::from_utf8(CONFIG).expect("config.json is UTF-8");
        let key = "\"hidden_dim\":";
        let start = text.find(key).expect("config.json declares hidden_dim") + key.len();
        let declared: usize = text[start..]
            .trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|digits| digits.parse().ok())
            .expect("hidden_dim is a number");

        let model = bundled().expect("the bundled model loads");
        assert_eq!(model.dim(), declared, "encoder width against config.json");
    }

    #[test]
    fn the_bundled_model_loads_from_embedded_bytes() {
        let model = bundled().expect("the bundled model loads");
        assert!(model.dim() > 0);
    }

    /// A real sentence gets a real vector: full width, and not the zero vector.
    ///
    /// The second half is what matters. Every "does it return something"
    /// assertion stays green if the encoder silently degrades to zeros, which is
    /// exactly what the no-token fallback in `Model::embed` would look like if
    /// it fired for ordinary text.
    #[test]
    fn ordinary_text_gets_a_full_width_non_zero_vector() {
        let model = bundled().expect("the bundled model loads");
        let vector = model
            .embed("a manufacturer of industrial fasteners in Sheffield")
            .expect("embed");
        assert_eq!(vector.len(), model.dim());
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.5, "expected a normalised vector, got norm {norm}");
    }

    /// Same string in, same vector out — the property the cache depends on.
    #[test]
    fn embedding_is_deterministic() {
        let model = bundled().expect("the bundled model loads");
        assert_eq!(
            model.embed("repeatable").expect("embed"),
            model.embed("repeatable").expect("embed")
        );
    }

    #[test]
    fn different_strings_get_different_vectors() {
        let model = bundled().expect("the bundled model loads");
        assert_ne!(
            model.embed("bicycle").expect("embed"),
            model.embed("sovereign debt").expect("embed")
        );
    }

    /// The empty-tokenisation contract, pinned.
    ///
    /// Empty and whitespace-only text yield a zero vector of the model's width.
    /// This is upstream behaviour, which is precisely why it is asserted here:
    /// a dependency bump that changed it would otherwise change the extension's
    /// SQL contract without anything reddening.
    #[test]
    fn text_with_no_tokens_gets_a_zero_vector_of_full_width() {
        let model = bundled().expect("the bundled model loads");
        // The last two are runic and alchemical characters: the vocabulary this
        // model was built on does not carry them, so the tokenizer produces only
        // unknown tokens and there is nothing to average.
        for text in [
            "",
            " ",
            "\t\n  ",
            "\u{16A0}\u{16A2}\u{16A6}",
            "\u{1F701}\u{1F702}\u{1F703}",
        ] {
            let vector = model.embed(text).expect("embed");
            assert_eq!(vector.len(), model.dim(), "width for {text:?}");
            assert!(
                vector.iter().all(|v| *v == 0.0),
                "expected a zero vector for {text:?}"
            );
        }
    }

    /// AC3 + AC5: the *token* boundary between "clipped" and "not", pinned in
    /// both directions with a probe built to actually show truncation.
    ///
    /// A text of one repeated token cannot show this: the mean of 512 copies of
    /// a vector equals the mean of 513 copies of the same vector, so a probe
    /// like that reports "not clipped" whichever token the encoder happens to
    /// drop and would pass against a limit of any size, correct or broken. This
    /// probe is `N` copies of one filler word plus one DISTINCT trailing marker
    /// word, so whether the marker reached the mean is a fact a byte-equality
    /// check on the two vectors can actually observe.
    ///
    /// The filler is `"ok "` — three characters per token — rather than a
    /// six-character word, on purpose: `MAX_TOKENS * median_token_length`
    /// characters (512 * 6 = 3072 here) is a *second*, independent boundary,
    /// and a filler word whose own characters-per-token happens to sit at the
    /// vocabulary median crosses it at almost the same repetition count as the
    /// token cap. An earlier version of this test used `"steel "` — six
    /// characters including the space, exactly the median — and 511 reps plus
    /// a nine-character marker landed at 3075 characters, three past the
    /// 3072-character cut. That test asserted "not clipped" for text that the
    /// character cut had already clipped, and passed, because nothing before
    /// this file's `Truncation` type checked the character cut at all. `"ok "`
    /// keeps this probe's total character count under 1600 at these
    /// repetition counts — nowhere near 3072 — so it is pinning the token
    /// boundary alone, uncontaminated by the other one.
    /// [`text_is_reported_clipped_by_the_character_cut_alone`] pins that other
    /// boundary the same way.
    #[test]
    fn a_long_text_is_reported_clipped_exactly_where_the_marker_stops_reaching_the_mean() {
        let model = bundled().expect("the bundled model loads");
        let filler = |tokens: usize| "ok ".repeat(tokens).trim_end().to_string();
        let filler_plus_marker = |tokens: usize| {
            let mut text = "ok ".repeat(tokens);
            text.push_str("marker");
            text
        };

        // 511 filler tokens + 1 marker = 512 tokens: exactly at the cap, so the
        // marker is the 512nd token and is still pooled. 1539 characters total
        // — far short of the 3072-character cut, so only the token cap is in
        // play here.
        let at_cap = filler_plus_marker(MAX_TOKENS - 1);
        assert!(
            !model.is_truncated(&at_cap),
            "{} tokens must not report clipped",
            MAX_TOKENS
        );
        assert_ne!(
            model.embed(&at_cap).expect("embed"),
            model.embed(&filler(MAX_TOKENS - 1)).expect("embed"),
            "the marker at position {MAX_TOKENS} must still move the mean"
        );

        // 512 filler tokens + 1 marker = 513 tokens: one past the cap, so the
        // marker is truncated away before pooling. 1542 characters — still far
        // short of 3072.
        let past_cap = filler_plus_marker(MAX_TOKENS);
        assert!(
            model.is_truncated(&past_cap),
            "{} tokens must report clipped",
            MAX_TOKENS + 1
        );
        assert_eq!(
            model.embed(&past_cap).expect("embed"),
            model.embed(&filler(MAX_TOKENS)).expect("embed"),
            "the marker at position {} must have been dropped before pooling",
            MAX_TOKENS + 1
        );
    }

    /// AC1 + AC3 + AC5: the *character* boundary between "clipped" and "not" —
    /// the one `embed_is_truncated` used to be structurally blind to.
    ///
    /// `model2vec_rs` cuts the raw string to `MAX_TOKENS * median_token_length`
    /// characters (3072, here) *before* tokenising, so text whose own
    /// characters-per-token sits above that median can lose content while
    /// still tokenising to far fewer than [`MAX_TOKENS`] ids — a token-count
    /// check alone, run on the untruncated text, would say "plenty of room"
    /// right up to the point content is already gone. `"internationalization "`
    /// is 21 characters for 2 tokens — 10.5 characters per token, well above
    /// the median of 6 — so it crosses the 3072-character cut at 147
    /// repetitions (3087 characters) while sitting at only 294 tokens, 218
    /// short of the token cap.
    #[test]
    fn text_is_reported_clipped_by_the_character_cut_alone() {
        let model = bundled().expect("the bundled model loads");
        let word = "internationalization ";
        let filler = |reps: usize| word.repeat(reps).trim_end().to_string();
        let filler_plus_marker = |reps: usize| {
            let mut text = word.repeat(reps);
            text.push_str("marker");
            text
        };

        // 145 reps of filler plus the marker is 3051 characters, well under
        // the 3072-character cut, and 291 tokens, well under MAX_TOKENS —
        // neither cap is in play, so the marker is pooled.
        let under_the_cut = filler_plus_marker(145);
        assert!(
            !model.is_truncated(&under_the_cut),
            "3051 characters, 291 tokens: must not report clipped"
        );
        assert_ne!(
            model.embed(&under_the_cut).expect("embed"),
            model.embed(&filler(145)).expect("embed"),
            "the marker must still move the mean"
        );

        // 147 reps of filler alone is already 3087 characters — past the
        // 3072-character cut — at only 294 tokens, 218 short of MAX_TOKENS.
        // The trailing marker sits entirely past the character cut, so it is
        // discarded before it is ever tokenised, and embedding it changes
        // nothing: the vector is identical to the filler alone.
        let over_the_cut = filler_plus_marker(147);
        assert!(
            model.is_truncated(&over_the_cut),
            "3093 characters, 295 tokens: must report clipped even though \
             token count alone is nowhere near {MAX_TOKENS}"
        );
        assert!(
            model.is_truncated(&filler(147)),
            "the filler alone, past the character cut, must also report clipped"
        );
        assert_eq!(
            model.embed(&over_the_cut).expect("embed"),
            model.embed(&filler(147)).expect("embed"),
            "the marker past the character cut must have been dropped before pooling"
        );
    }

    /// AC4 at the engine: `Model::embed` now calls `encode_with_args` directly
    /// with this crate's own [`MAX_TOKENS`] rather than model2vec-rs's
    /// `encode_single` convenience wrapper (which hard-codes its own `512`).
    /// Pinned against the old call path so that refactor is provably a no-op.
    #[test]
    fn embed_matches_the_upstream_convenience_wrapper_it_replaced() {
        let model = bundled().expect("the bundled model loads");

        // A text made of one token repeated cannot exercise this: the mean of
        // 511 copies of a vector equals the mean of 512 copies of it, so a
        // wrong cap on one side would still pass. This text has a token that
        // is inside one cap and outside the other — a marker as the 512th
        // token — so a one-token disagreement between the two call paths is
        // something this comparison can actually see. It also happens to sit
        // 3 characters past the 3072-character cut, so it exercises the
        // character-cut path as well as the token-cut path.
        let mut boundary_marker = "steel ".repeat(MAX_TOKENS - 1);
        boundary_marker.push_str("logistics");

        // The reachable class the round's defect was found in: text whose
        // characters-per-token sits above the vocabulary median, so the
        // character cut fires while the token count is still far under
        // MAX_TOKENS. `model.inner.encode_single` runs the unmodified upstream
        // path — both cuts, on the original call shape — so equality here is
        // the proof that reading `Truncation::surviving_prefix` back through a
        // fresh `encode_with_args(None)` reproduces it exactly, not just for
        // the cases this crate's own truncation logic was designed against.
        let dense_word = "internationalization ";
        let just_under_the_char_cut = dense_word.repeat(146);
        let just_over_the_char_cut = dense_word.repeat(147);
        let far_over_the_char_cut = dense_word.repeat(300);

        for text in [
            "",
            "a foundry casting valve bodies",
            &"steel ".repeat(600),
            boundary_marker.as_str(),
            just_under_the_char_cut.as_str(),
            just_over_the_char_cut.as_str(),
            far_over_the_char_cut.as_str(),
        ] {
            assert_eq!(
                model.embed(text).expect("embed"),
                conform(model.inner.encode_single(text), model.dim()).expect("conform"),
                "{} chars, {:?} preview",
                text.chars().count(),
                &text[..text.len().min(40)]
            );
        }
    }

    /// Nothing to average is not clipped: a text with no in-vocabulary tokens
    /// has nothing dropped, it simply never had anything past the limit.
    #[test]
    fn text_with_no_tokens_is_not_reported_clipped() {
        let model = bundled().expect("the bundled model loads");
        for text in ["", "   ", "\u{16A0}\u{16A2}\u{16A6}"] {
            assert!(!model.is_truncated(text), "{text:?}");
        }
    }

    /// Ordinary short text is not clipped, and a text built to be many times
    /// past the cap is — the coarse sanity check either side of the pinned
    /// boundary above.
    #[test]
    fn ordinary_text_is_not_clipped_and_a_much_longer_text_is() {
        let model = bundled().expect("the bundled model loads");
        assert!(!model.is_truncated("a manufacturer of industrial fasteners in Sheffield"));
        assert!(model.is_truncated(&"steel ".repeat(MAX_TOKENS * 3)));
    }

    /// The model key is a content address: flipping one byte of any of the three
    /// assets moves it.
    ///
    /// This calls `model_key` itself rather than recomputing the same fields
    /// beside it. A version that recomputed them would still pass with an asset
    /// dropped from the production hash, because the test's own copy would keep
    /// including it — which is what this test used to do.
    #[test]
    fn the_model_key_covers_every_asset_byte() {
        let live = model_key(TOKENIZER, WEIGHTS, CONFIG);
        assert_eq!(hex(&live).len(), 64);

        let flip_last = |bytes: &[u8]| {
            let mut copy = bytes.to_vec();
            let last = copy.len() - 1;
            copy[last] ^= 0x01;
            copy
        };

        assert_ne!(
            live,
            model_key(&flip_last(TOKENIZER), WEIGHTS, CONFIG),
            "tokenizer"
        );
        assert_ne!(
            live,
            model_key(TOKENIZER, &flip_last(WEIGHTS), CONFIG),
            "weights"
        );
        assert_ne!(
            live,
            model_key(TOKENIZER, WEIGHTS, &flip_last(CONFIG)),
            "config"
        );
    }

    /// The key a loaded model reports is the key its own asset bytes derive.
    #[test]
    fn the_loaded_model_reports_the_key_its_assets_derive() {
        let model = bundled().expect("the bundled model loads");
        assert_eq!(model.key(), &model_key(TOKENIZER, WEIGHTS, CONFIG));
    }

    /// A full-width vector passes through untouched.
    #[test]
    fn conform_leaves_a_full_width_vector_alone() {
        assert_eq!(
            conform(vec![1.0, 2.0, 3.0, 4.0], 4),
            Ok(vec![1.0, 2.0, 3.0, 4.0])
        );
    }

    /// Nothing to average becomes the zero vector at full width.
    #[test]
    fn conform_turns_an_empty_vector_into_a_full_width_zero_vector() {
        assert_eq!(conform(vec![], 4), Ok(vec![0.0, 0.0, 0.0, 0.0]));
    }

    /// Any other width is a bug in the encoder and is reported, not padded.
    ///
    /// Silent padding here would turn a dimension mismatch into a column of
    /// plausible-looking zeros, which nothing downstream could distinguish from
    /// text that genuinely had no tokens.
    #[test]
    fn conform_reports_any_other_width_rather_than_padding_it() {
        let reported = conform(vec![1.0, 2.0, 3.0], 4).expect_err("a short vector is an error");
        assert!(
            reported.contains('3') && reported.contains('4'),
            "{reported}"
        );
    }
}
