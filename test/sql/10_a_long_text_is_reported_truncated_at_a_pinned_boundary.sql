-- AC1, AC3, AC4, AC5: `embed_is_truncated` is a SQL-level predicate an analyst
-- can put in a WHERE clause and a count(*), the boundary it reports is pinned
-- in both directions on BOTH of the limits `embed` applies — a token count and
-- a character count ahead of it — and `embed`'s own contract is unchanged by
-- its presence.
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

-- The pinned boundary. `repeat('ok ', n)` gives `n` filler tokens (measured:
-- one word, one token, in this vocabulary); `marker` is one DISTINCT token
-- appended directly after, with no boundary token count ambiguity.
--
-- `ok ` rather than `steel `: `embed` truncates on TWO independent limits, a
-- token count and a character count, and `steel ` is six characters — the
-- exact median token length in this vocabulary, which is also what the
-- character limit is measured in units of. 511 reps of `steel ` plus a
-- nine-character marker lands three characters past that second limit, so a
-- boundary built from it is pinning the character cut, not the token cut, and
-- an earlier version of this file asserted "not truncated" for text the
-- character cut had already clipped. `ok ` is three characters per token, so
-- even 512 reps plus the marker stays under 1600 characters — nowhere near
-- the character limit — and this table pins the token boundary alone.
CREATE TABLE boundary AS SELECT * FROM (VALUES
    -- 511 filler + 1 marker = 512 tokens: exactly at the cap.
    (511, repeat('ok ', 511) || 'marker', rtrim(repeat('ok ', 511))),
    -- 512 filler + 1 marker = 513 tokens: one past the cap.
    (512, repeat('ok ', 512) || 'marker', rtrim(repeat('ok ', 512)))
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

-- AC1 + AC3: the *character* boundary, pinned the same way — the direction
-- `embed_is_truncated` used to be structurally blind to. `embed` cuts the raw
-- string to 3072 characters before it ever tokenises, so text whose own
-- characters-per-token sits above this vocabulary's median (6) can lose
-- content while its full, untruncated token count is nowhere near 512.
-- `internationalization ` is 21 characters for 2 tokens — crosses the
-- 3072-character cut at 147 reps (3087 characters) while sitting at only 294
-- tokens.
CREATE TABLE char_boundary AS SELECT * FROM (VALUES
    -- 146 reps = 3066 characters: under the character cut, and only 292
    -- tokens — nowhere near the token cap either.
    (146, repeat('internationalization ', 146) || 'marker', rtrim(repeat('internationalization ', 146))),
    -- 147 reps = 3087 characters: past the character cut, at 294 tokens —
    -- still 218 short of the token cap.
    (147, repeat('internationalization ', 147) || 'marker', rtrim(repeat('internationalization ', 147)))
) AS t(filler_reps, with_marker, filler_only);

SELECT must('under the character cut, with far fewer than 512 tokens, is not reported truncated',
    (SELECT embed_is_truncated(with_marker) FROM char_boundary WHERE filler_reps = 146) = false);
SELECT must('under the character cut the marker still reaches the mean',
    (SELECT embed(with_marker) FROM char_boundary WHERE filler_reps = 146)
    IS DISTINCT FROM
    (SELECT embed(filler_only) FROM char_boundary WHERE filler_reps = 146));

SELECT must('past the character cut, still far fewer than 512 tokens, is reported truncated',
    (SELECT embed_is_truncated(with_marker) FROM char_boundary WHERE filler_reps = 147) = true);
SELECT must('past the character cut the marker was dropped before pooling, though token count alone gave no warning',
    (SELECT embed(with_marker) FROM char_boundary WHERE filler_reps = 147)
    IS NOT DISTINCT FROM
    (SELECT embed(filler_only) FROM char_boundary WHERE filler_reps = 147));

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
