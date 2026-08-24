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

## Status

Early. The extension is being built; nothing is published to the community registry yet.

## Licence

MIT. The embedding model is a third-party Model2Vec release and carries its own licence, reproduced with the bundled weights.
