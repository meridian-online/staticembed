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

/// Domain tag mixed into the model key so the digest cannot be confused with a
/// plain SHA-256 of any one asset.
const MODEL_KEY_DOMAIN: &[u8] = b"staticembed/model-key/v1";

/// A loaded static embedding model.
pub struct Model {
    inner: StaticModel,
    dim: usize,
    key: [u8; 32],
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

        let mut hasher = Sha256::new();
        hasher.update(MODEL_KEY_DOMAIN);
        hasher.update(MODEL_ID.as_bytes());
        hasher.update([0u8]);
        hasher.update(MODEL_REVISION.as_bytes());
        hasher.update([0u8]);
        hasher.update(TOKENIZER);
        hasher.update(WEIGHTS);
        hasher.update(CONFIG);
        let key: [u8; 32] = hasher.finalize().into();

        Ok(Model { inner, dim, key })
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
    /// The `dim` fallback below returns the same value the encoder returns for
    /// no-token input, so it changes no embedding — it only guarantees the
    /// width when the encoder returns nothing at all.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let vector = self.inner.encode_single(text);
        if vector.len() == self.dim {
            vector
        } else {
            vec![0.0_f32; self.dim]
        }
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
        let vector = model.embed("a manufacturer of industrial fasteners in Sheffield");
        assert_eq!(vector.len(), model.dim());
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 0.5, "expected a normalised vector, got norm {norm}");
    }

    /// Same string in, same vector out — the property the cache depends on.
    #[test]
    fn embedding_is_deterministic() {
        let model = bundled().expect("the bundled model loads");
        assert_eq!(model.embed("repeatable"), model.embed("repeatable"));
    }

    #[test]
    fn different_strings_get_different_vectors() {
        let model = bundled().expect("the bundled model loads");
        assert_ne!(model.embed("bicycle"), model.embed("sovereign debt"));
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
            let vector = model.embed(text);
            assert_eq!(vector.len(), model.dim(), "width for {text:?}");
            assert!(
                vector.iter().all(|v| *v == 0.0),
                "expected a zero vector for {text:?}"
            );
        }
    }

    /// The model key is a content address: it moves when the assets move.
    #[test]
    fn the_model_key_covers_the_asset_bytes() {
        let model = bundled().expect("the bundled model loads");
        let live = model.key_hex();
        assert_eq!(live.len(), 64);

        // Recompute with one byte of the weights flipped. Nothing else changes.
        let mut mutated = WEIGHTS.to_vec();
        let last = mutated.len() - 1;
        mutated[last] ^= 0x01;
        let mut hasher = Sha256::new();
        hasher.update(MODEL_KEY_DOMAIN);
        hasher.update(MODEL_ID.as_bytes());
        hasher.update([0u8]);
        hasher.update(MODEL_REVISION.as_bytes());
        hasher.update([0u8]);
        hasher.update(TOKENIZER);
        hasher.update(&mutated);
        hasher.update(CONFIG);
        assert_ne!(
            live,
            hex(&hasher.finalize()),
            "one flipped weight byte must change the model key"
        );
    }
}
