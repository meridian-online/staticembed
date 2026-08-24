-- AC3, from this side of the boundary: staticembed is its own extension with
-- its own surface. finetype is a different repo and a different artifact, and
-- nothing here uses its prefix or its type contract.
--
-- The surface is derived, not asserted from memory: the prelude snapshots
-- duckdb_functions() before LOAD, so the delta below is exactly what this
-- extension registered. Adding a fifth function reddens this file.

CREATE TABLE registered AS
    SELECT DISTINCT function_name FROM duckdb_functions()
    WHERE function_name NOT IN (SELECT function_name FROM staticembed_baseline_functions)
      AND function_name <> 'must';

SELECT must('the extension registers four functions',
    (SELECT count(*) FROM registered) = 4);

SELECT must('and they are exactly the documented four',
    (SELECT list_sort(list(function_name)) FROM registered)
    = ['embed', 'staticembed_cache_clear', 'staticembed_cache_stats', 'staticembed_version']);

-- The card does not license nearest-neighbour lookup: the measured position of
-- this model is that a map from its vectors keeps the clusters and loses the
-- neighbourhoods. A function that invited "show me the rows most like this one"
-- would promise what the model does not deliver.
SELECT must('no similarity or nearest-neighbour function is registered',
    (SELECT count(*) FROM registered
     WHERE regexp_matches(function_name, '(?i)similar|neighbou?r|nearest|knn|distance|match|rank')) = 0);

SELECT must('nothing here takes the ft_ prefix that belongs to another extension',
    (SELECT count(*) FROM registered WHERE starts_with(function_name, 'ft_')) = 0);

SELECT must('every registered function is a scalar',
    (SELECT count(*) FROM duckdb_functions()
     WHERE function_name IN (SELECT function_name FROM registered)
       AND function_type <> 'scalar') = 0);
