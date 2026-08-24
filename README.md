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

It also does not read word order. A Model2Vec vector is the mean of its token vectors, so `embed('valve bodies')` and `embed('bodies valve')` are the same vector. Repetition does count — a word twice pulls the mean toward it — but any phrase and its shuffle land in the same place.

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

### What counts as the same string

The bundled tokenizer lowercases, strips accents and ignores surrounding whitespace, so `embed('Steel')`, `embed('steel')` and `embed('  steel  ')` are the same vector, and `embed('café')` matches `embed('cafe' || chr(769))`. You do not have to normalise a column before embedding it. Word order still matters, and different words still give different vectors.

The cache keys on the exact input bytes rather than on the tokenizer's folded form, so those variants do occupy separate cache entries. That is deliberate: reproducing a dependency's normalisation in the cache would mean a tokenizer bump quietly changing which inputs share an entry, and the failure would be a vector returned for text nobody embedded.

### Repeating a query does not re-embed, up to a stated number of distinct values

Vectors are cached against the exact input bytes and a digest of the bundled model's own files, so re-running a query re-embeds nothing — **for as many distinct values as the cache holds**, and a future build with different weights cannot serve you the old ones. `staticembed_cache_stats()` is how you see which case you are in rather than guessing from how long the query took:

```sql
SELECT staticembed_cache_clear();
CREATE TABLE v AS SELECT embed(description) FROM corpus;
SELECT staticembed_cache_stats();   -- uncached 0 means the column fitted, and
                                    -- then encoded is one per distinct value
CREATE TABLE w AS SELECT embed(description) FROM corpus;
SELECT staticembed_cache_stats();   -- encoded unchanged, if uncached was 0
```

**The bound.** The cache spends a fixed memory budget, so it holds `capacity` vectors and no more; `staticembed_cache_stats().capacity` reports the figure for your build, and it falls as the model's vector width rises. Once it is full it **stops admitting new values rather than evicting old ones**, and `uncached` counts every lookup it turned away.

That choice is the whole behaviour above the bound, so it is worth being plain about. A column with more distinct values than `capacity` is served for `capacity` of them on a re-run and re-embeds the rest — a hit rate of `capacity / distinct`, holding steady as the column grows. Under any recency-ordered policy instead — LRU, FIFO, or the two generations this extension shipped with first — a repeated scan evicts every value exactly before it is next wanted and the hit rate is **zero**, which is a cliff rather than a slope and is much worse than it sounds.

Above the bound the guarantee is weaker in a second way as well: a value the cache turned away is re-embedded for every row that carries it, and two threads meeting the same turned-away value will each embed it. Below the bound neither happens — one encode per distinct value, whatever the thread count.

What it costs is adaptivity: the values kept are the ones seen first in the session, so if you move on to a different column the cache stays full of the old one. `SELECT staticembed_cache_clear();` empties it, and a non-zero `uncached` is the sign that it is time.

## Building it

Needs a Rust toolchain, Python 3 for the packaging and test scripts, and the `duckdb` CLI for the SQL tests.

```
make check           # formatting, clippy, Rust tests, the artifact, and the SQL tests
make extension       # just the loadable artifact, in build/
make mutation-check  # break the code on purpose and require the tests to notice
```

`make check` is what CI runs. `make mutation-check` is not in CI, because each of its SQL mutations rebuilds the release binary; it applies a table of deliberate defects one at a time and fails unless the test named against each one reddens, which is the difference between a suite that is green and a suite that is watching.

The model is compiled into the binary, so the artifact is large — most of it is weights. Loading it needs `duckdb -unsigned` until a signed build exists in the community registry:

```sql
LOAD 'build/staticembed.duckdb_extension';
```


## Status

Early. The extension builds, loads and answers queries; nothing is published to the community registry yet, so the `INSTALL ... FROM community` line at the top of this page does not work today. Build it yourself with `make extension` in the meantime.

## The model

`minishlab/potion-base-8M`, a Model2Vec static embedding model, taken from its published release at a pinned revision and compiled into the binary. Its files, their checksums and the revision they came from are in [`models/potion-base-8M/SOURCE.md`](models/potion-base-8M/SOURCE.md), and a test recomputes those checksums so a swapped asset reddens rather than shipping. The quality position above was measured on this model; a different one would need the measurement redone.

## Licence

This repository is MIT; the `LICENSE` file at its root is that licence and covers the code here.

The bundled embedding model is a third-party Model2Vec release. Its publisher **declares** it MIT, in the model card at [`models/potion-base-8M/MODEL_CARD.md`](models/potion-base-8M/MODEL_CARD.md) — the frontmatter and the citation both say so. The upstream repository carries no `LICENSE` file at the pinned revision, so **no MIT text or copyright line for the model is reproduced here**, because there is none to copy. If you need the licence in hand rather than declared, take it up with the publisher before redistributing the weights.
