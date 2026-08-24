# staticembed

A DuckDB scalar function that turns a string into a vector, using a static embedding model bundled in the extension binary. No API key, no network call, no per-row bill.

```sql
INSTALL staticembed FROM community;
LOAD staticembed;

SELECT description, embed(description) AS v
FROM read_parquet('corpus.parquet')
WHERE industry = 'manufacturing'
LIMIT 100;
```

## Why this exists

Every other route to an embedding inside DuckDB is either a transformer forward pass or a call to a hosted provider. Over a large text column the first is slow enough that people leave SQL to do it, and the second is an invoice you find out about afterwards. A static embedding is neither: it is a token lookup and a mean, so it is bounded, local, and free at the margin.

Being a scalar is the point. It composes with `WHERE` and `LIMIT`, so you can embed a filtered subset rather than a whole table, and it behaves the same over a local Parquet file as over a remote one.

## What it is not

**It is not a drop-in replacement for a hosted transformer, and the difference is measured rather than hedged.** A static embedding has no contextual attention. On the evidence we have, a two-dimensional map built from these vectors recovers most of the cluster structure a hosted transformer recovers — the regions of the map mean something. What it does not preserve is which specific rows sit next to which: only a minority of any point's nearest neighbours survive the swap.

So this is built for looking at the shape of a corpus, and it is not built for "show me the rows most like this one". If nearest-neighbour lookup is what you need, this is the wrong tool, and we would rather say so here than have you find out from your results.

Figures, corpora and method are published with the measurement in [meridian-online/finetype](https://github.com/meridian-online/finetype).

## The SQL surface

Four functions, and `embed` is the one you came for.

| function | returns | what it is for |
|---|---|---|
| `embed(text VARCHAR)` | `FLOAT[]` | the vector for one string |
| `staticembed_version()` | `VARCHAR` | which build, which model, which vector width |
| `staticembed_cache_stats()` | `STRUCT(hits, misses, encoded, entries, capacity)` | what the cache has been doing |
| `staticembed_cache_clear()` | `BIGINT` | drop the cached vectors; returns how many |

There is no similarity or nearest-neighbour function, deliberately. See *What it is not* above.

### NULL, and text with nothing in it

`embed(NULL)` is `NULL`. Text that tokenises to nothing — the empty string, whitespace, a string of characters the vocabulary does not carry — is a **zero vector of full width**, because the mean over zero tokens is zero. The two are different on purpose: a missing value is not the same as a value that carries no signal, and only one of them should disappear from a `WHERE ... IS NOT NULL`.

If you want the single behaviour a text pipeline usually gives you, ask for it:

```sql
SELECT embed(coalesce(description, '')) FROM corpus;
```

### Repeating a query does not re-embed

Vectors are cached against the exact input bytes and a digest of the bundled model's own files, so re-running a query over unchanged input costs nothing, and a future build with different weights cannot serve you the old vectors. `staticembed_cache_stats().encoded` counts how many times the encoder has actually run since the last clear, which is how you can see it for yourself:

```sql
SELECT staticembed_cache_clear();
CREATE TABLE v AS SELECT embed(description) FROM corpus;
SELECT staticembed_cache_stats().encoded;   -- one per distinct value
CREATE TABLE w AS SELECT embed(description) FROM corpus;
SELECT staticembed_cache_stats().encoded;   -- unchanged
```

The cache is bounded rather than unlimited: it holds two generations of `capacity` entries, and a value that keeps being asked for is kept. `staticembed_cache_stats().capacity` reports the size in force.

## Building it

Needs a Rust toolchain, Python 3 for the packaging and test scripts, and the `duckdb` CLI for the SQL tests.

```
make check        # formatting, clippy, Rust tests, the artifact, and the SQL tests
make extension    # just the loadable artifact, in build/
```

The model is compiled into the binary, so the artifact is large — most of it is weights. Loading it needs `duckdb -unsigned` until a signed build exists in the community registry:

```sql
LOAD 'build/staticembed.duckdb_extension';
```


## Status

Early. The extension builds, loads and answers queries; nothing is published to the community registry yet, so the `INSTALL ... FROM community` line at the top of this page does not work today. Build it yourself with `make extension` in the meantime.

## The model

`minishlab/potion-base-8M`, a Model2Vec static embedding model, taken from its published release at a pinned revision and compiled into the binary. Its files, their checksums and the revision they came from are in [`models/potion-base-8M/SOURCE.md`](models/potion-base-8M/SOURCE.md), and a test recomputes those checksums so a swapped asset reddens rather than shipping. The quality position above was measured on this model; a different one would need the measurement redone.

## Licence

MIT. The embedding model is a third-party Model2Vec release under its own MIT licence, reproduced with the bundled weights in [`models/potion-base-8M/MODEL_CARD.md`](models/potion-base-8M/MODEL_CARD.md).
