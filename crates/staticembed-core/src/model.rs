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
/// tokenises (after the encoder's own char-level pre-truncation and
/// out-of-vocabulary filter) to more ids than this is truncated, and the
/// excess never reaches the mean.
///
/// This is `model2vec_rs::StaticModel::encode`'s own default, `Some(512)` —
/// declared here, and passed explicitly by [`Model::embed`], rather than left
/// as the literal inside that call, so this crate has exactly one place that
/// says "512" instead of two that are expected to agree. [`Model::is_truncated`]
/// is the other reader of this constant; a test pins that the two stay in
/// step at the boundary.
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
    truncation: TruncationProbe,
}

/// A second, independent reader of the bundled tokenizer, used only to answer
/// "would `embed` truncate this text" without paying for a full encode.
///
/// `model2vec_rs::StaticModel` parses its own tokenizer at load and keeps it
/// private, with no accessor — so answering this question needs a tokenizer of
/// its own. It is loaded from the identical `TOKENIZER` bytes [`Model`] itself
/// loads from, so the two agree on what a token is.
///
/// This deliberately does **not** reproduce `encode_with_args`'s char-level
/// pre-truncation (to `max_tokens * median_token_length` characters, a private
/// optimisation with no public accessor for the length it uses). A first
/// version did, recomputing its own copy of that median from the vocabulary —
/// and for a text built of one token repeated, where the per-token character
/// count lands exactly on that median, the two truncations coincide: the
/// probe's own pre-cut already lands on exactly [`MAX_TOKENS`] ids, so the
/// over-the-limit count it goes on to check can never exceed the limit it is
/// checking against, and the probe reports "not clipped" for text of any
/// length. Tokenising the whole string and counting is slower for a
/// pathologically long text, but it cannot mask itself this way, and the
/// evidence behind this card found no text where the character cap fires
/// before the token cap does.
struct TruncationProbe {
    tokenizer: Tokenizer,
    /// The id `model2vec_rs` drops from a token list before truncating and
    /// pooling it. Recomputed here the same way its own (private) metadata
    /// step does: there is no accessor for it either.
    unk_token_id: Option<usize>,
}

impl TruncationProbe {
    fn from_tokenizer_bytes(bytes: &[u8]) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_bytes(bytes)
            .map_err(|e| format!("the truncation probe's tokenizer did not load: {e}"))?;

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

        Ok(Self {
            tokenizer,
            unk_token_id,
        })
    }

    /// True if embedding `text` drops tokens: applying the same
    /// out-of-vocabulary filter `encode_with_args` applies to the whole string
    /// leaves more than [`MAX_TOKENS`] ids, so the real call truncates the list
    /// before pooling.
    ///
    /// A `false` for text with no in-vocabulary tokens at all — the empty
    /// string, whitespace, symbols outside the vocabulary — is correct, not a
    /// false negative: nothing was discarded, there was simply nothing there.
    fn truncates(&self, text: &str) -> bool {
        let Ok(encoding) = self.tokenizer.encode_fast(text, false) else {
            // A failure here would also fail inside `embed`, which surfaces it
            // as an ordinary error. This probe backs a SQL predicate rather
            // than a fallible call, so it degrades to "not observed to
            // truncate" instead of erroring a scan.
            return false;
        };

        let count = match self.unk_token_id {
            Some(unk) => encoding
                .get_ids()
                .iter()
                .filter(|&&id| id as usize != unk)
                .count(),
            None => encoding.get_ids().len(),
        };
        count > MAX_TOKENS
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
        let truncation = TruncationProbe::from_tokenizer_bytes(TOKENIZER)?;

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
        let sentence = [text.to_string()];
        let vector = self
            .inner
            .encode_with_args(&sentence, Some(MAX_TOKENS), 1)
            .into_iter()
            .next()
            .unwrap_or_default();
        conform(vector, self.dim)
    }

    /// Whether embedding `text` discards content: `text` tokenises to more
    /// than [`MAX_TOKENS`] ids, so [`Model::embed`] truncates the excess before
    /// pooling and the vector it returns does not reflect all of `text`.
    ///
    /// The vector `embed` returns is full width and unit norm whether or not
    /// this is true — nothing about it says content was dropped, which is the
    /// whole reason to ask first.
    pub fn is_truncated(&self, text: &str) -> bool {
        self.truncation.truncates(text)
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

    /// AC3 + AC5: the boundary between "clipped" and "not", pinned in both
    /// directions with a probe built to actually show truncation.
    ///
    /// A text of one repeated token cannot show this: the mean of 512 copies of
    /// a vector equals the mean of 513 copies of the same vector, so a probe
    /// like that reports "not clipped" whichever token the encoder happens to
    /// drop and would pass against a limit of any size, correct or broken. This
    /// probe is `N` copies of one filler word plus one DISTINCT trailing marker
    /// word, so whether the marker reached the mean is a fact a byte-equality
    /// check on the two vectors can actually observe.
    #[test]
    fn a_long_text_is_reported_clipped_exactly_where_the_marker_stops_reaching_the_mean() {
        let model = bundled().expect("the bundled model loads");
        let filler = |tokens: usize| "steel ".repeat(tokens).trim_end().to_string();
        let filler_plus_marker = |tokens: usize| {
            let mut text = "steel ".repeat(tokens);
            text.push_str("logistics");
            text
        };

        // 511 filler tokens + 1 marker = 512 tokens: exactly at the cap, so the
        // marker is the 512nd token and is still pooled.
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
        // marker is truncated away before pooling.
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

    /// AC4 at the engine: `Model::embed` now calls `encode_with_args` directly
    /// with this crate's own [`MAX_TOKENS`] rather than model2vec-rs's
    /// `encode_single` convenience wrapper (which hard-codes its own `512`).
    /// Pinned against the old call path so that refactor is provably a no-op.
    #[test]
    fn embed_matches_the_upstream_convenience_wrapper_it_replaced() {
        let model = bundled().expect("the bundled model loads");
        for text in ["", "a foundry casting valve bodies", &"steel ".repeat(600)] {
            assert_eq!(
                model.embed(text).expect("embed"),
                conform(model.inner.encode_single(text), model.dim()).expect("conform"),
                "{text:?}"
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
