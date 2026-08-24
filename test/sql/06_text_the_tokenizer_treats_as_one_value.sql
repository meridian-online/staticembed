-- What counts as the same string.
--
-- The bundled tokenizer lowercases, strips accents and drops surrounding
-- whitespace, so these are one value to the model. That is a SQL contract a
-- caller can lean on — `embed(name)` matches `embed(NAME)` without the caller
-- normalising first — and it is a property of the bundled asset rather than of
-- this repo's code, which is exactly why it is asserted rather than assumed.

SELECT must('case does not change the vector',
    embed('steel') IS NOT DISTINCT FROM embed('STEEL'));

SELECT must('surrounding whitespace does not change the vector',
    embed('steel') IS NOT DISTINCT FROM embed('   steel   '));

SELECT must('composed and decomposed accents give the same vector',
    embed('caf' || chr(233)) IS NOT DISTINCT FROM embed('cafe' || chr(769)));

-- The folding is the tokenizer's, not a blanket "everything is the same".
SELECT must('different words still give different vectors',
    embed('steel') IS DISTINCT FROM embed('copper'));

-- Word order does NOT change the vector, and that is the model rather than a
-- bug: a Model2Vec vector is the mean of its token vectors, and a mean does not
-- know what order it was taken in. Pinned so nobody reaches for this to compare
-- phrases where the order is the meaning.
SELECT must('word order does not change the vector, because the pool is a mean',
    embed('valve bodies') IS NOT DISTINCT FROM embed('bodies valve'));

SELECT must('a repeated word does change the vector, because the mean is weighted by count',
    embed('steel steel copper') IS DISTINCT FROM embed('steel copper'));

-- Those pairs are still separate cache entries: the key is the exact bytes,
-- because reproducing the tokenizer's normalisation in the cache would mean a
-- tokenizer bump silently changing which inputs share an entry.
SELECT staticembed_cache_clear();
CREATE TABLE two AS SELECT embed('steel') AS a, embed('STEEL') AS b;
SELECT must('one vector, two cache entries, two encodes',
    (SELECT staticembed_cache_stats().entries) = 2
    AND (SELECT staticembed_cache_stats().encoded) = 2);
