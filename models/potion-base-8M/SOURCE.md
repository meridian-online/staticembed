# Bundled model

`minishlab/potion-base-8M`, a Model2Vec static embedding model, taken from the published Hugging Face release and embedded in the extension binary at build time. Licence: MIT (see `MODEL_CARD.md`).

Source: `https://huggingface.co/minishlab/potion-base-8M`
Revision: `bf8b056651a2c21b8d2565580b8569da283cab23`

Three files are bundled — the weights, the tokenizer, and the config. The config is here because `model2vec_rs::model::StaticModel::from_bytes` reads the model's `normalize` flag from it; taking it from the release keeps that flag the model's own value rather than one this repo asserts.

| file | sha256 |
|---|---|
| `model.safetensors` | `f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2` |
| `tokenizer.json` | `e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50` |
| `config.json` | `2a6ac0e9aaa356a68a5688070db78fc3a464fefe85d2f06a1905ce3718687553` |

`cargo test -p staticembed-core bundled_asset_digests_match_the_pinned_release` recomputes all three and compares them against the constants in `crates/staticembed-core/src/model.rs`, so a swapped or truncated asset reddens rather than being silently embedded.
