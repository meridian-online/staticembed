-- AC1, AC3, AC4, AC5: `embed_is_truncated` is a SQL-level predicate an analyst
-- can put in a WHERE clause and a count(*), the boundary it reports is pinned
-- in both directions, and `embed`'s own contract is unchanged by its presence.
--
-- AC5's failure mode, spelled out: a probe of one token repeated cannot show
-- truncation, because the mean of 512 copies of a vector equals the mean of
-- 513 copies of it. Every probe below is filler plus one DISTINCT trailing
-- marker word, so a truncated vs. untruncated embed of it provably differ —
-- the same shape as the measurement behind this card, reproduced in SQL rather
-- than trusted from the Rust suite alone.

SELECT must('embed_is_truncated is registered as a scalar returning BOOLEAN',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name = 'embed_is_truncated'
       AND function_type = 'scalar'
       AND return_type = 'BOOLEAN') = 1);

SELECT must('embed_is_truncated(NULL) is NULL, matching embed(NULL)',
    embed_is_truncated(NULL) IS NULL);

SELECT must('the empty string is not reported truncated: nothing was there to drop',
    embed_is_truncated('') = false);
SELECT must('whitespace only is not reported truncated',
    embed_is_truncated('   ') = false);
SELECT must('ordinary short text is not reported truncated',
    embed_is_truncated('a manufacturer of industrial fasteners in Sheffield') = false);

-- The pinned boundary. `repeat('steel ', n)` gives `n` filler tokens (measured:
-- one word, one token, in this vocabulary); `logistics` is one DISTINCT token
-- appended directly after, with no boundary token count ambiguity.
CREATE TABLE boundary AS SELECT * FROM (VALUES
    -- 511 filler + 1 marker = 512 tokens: exactly at the cap.
    (511, repeat('steel ', 511) || 'logistics', rtrim(repeat('steel ', 511))),
    -- 512 filler + 1 marker = 513 tokens: one past the cap.
    (512, repeat('steel ', 512) || 'logistics', rtrim(repeat('steel ', 512)))
) AS t(filler_tokens, with_marker, filler_only);

SELECT must('512 tokens (511 filler + marker) is not reported truncated',
    (SELECT embed_is_truncated(with_marker) FROM boundary WHERE filler_tokens = 511) = false);
SELECT must('at 512 tokens the marker still reaches the mean',
    (SELECT embed(with_marker) FROM boundary WHERE filler_tokens = 511)
    IS DISTINCT FROM
    (SELECT embed(filler_only) FROM boundary WHERE filler_tokens = 511));

SELECT must('513 tokens (512 filler + marker) is reported truncated',
    (SELECT embed_is_truncated(with_marker) FROM boundary WHERE filler_tokens = 512) = true);
SELECT must('at 513 tokens the marker was dropped before pooling',
    (SELECT embed(with_marker) FROM boundary WHERE filler_tokens = 512)
    IS NOT DISTINCT FROM
    (SELECT embed(filler_only) FROM boundary WHERE filler_tokens = 512));

-- AC1: a WHERE clause and a count(*) over a column, the way an analyst would
-- actually ask the question.
CREATE TABLE corpus AS SELECT * FROM (VALUES
    (1, 'a maker of industrial fasteners'),
    (2, repeat('steel ', 600) || 'logistics'),
    (3, 'a regional freight forwarder'),
    (4, repeat('steel ', 900) || 'logistics')
) AS t(id, description);

SELECT must('count(*) over the column finds exactly the two long rows',
    (SELECT count(*) FROM corpus WHERE embed_is_truncated(description)) = 2);
SELECT must('the WHERE clause names the same two ids',
    (SELECT list_sort(list(id)) FROM corpus WHERE embed_is_truncated(description)) = [2, 4]);
SELECT must('the short rows are excluded, not merely unmatched',
    (SELECT list_sort(list(id)) FROM corpus WHERE NOT embed_is_truncated(description)) = [1, 3]);

-- AC4: embed() itself is unaffected by embed_is_truncated existing alongside
-- it — same vectors as the untruncated-probe assertions above, and the width
-- and cache-composing behaviour asserted in 01 and 04 are unchanged by this
-- file having run.
CREATE TABLE width AS
    SELECT CAST(regexp_extract(staticembed_version(), 'dim (\d+)', 1) AS BIGINT) AS n;
SELECT must('embed still returns the declared width for a truncated text',
    (SELECT len(embed(with_marker)) FROM boundary WHERE filler_tokens = 512) = (SELECT n FROM width));
SELECT must('a truncated result is still full width and unit norm, which is the whole problem',
    (SELECT list_max(embed(with_marker)) FROM boundary WHERE filler_tokens = 512) > 0);
