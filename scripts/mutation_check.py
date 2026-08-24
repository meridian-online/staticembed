#!/usr/bin/env python3
"""Break the code on purpose and require the named test to redden.

A test that passes against working code AND against deliberately broken code is
testing nothing, and reading it again will not tell you which kind you have.
This applies each mutation below to a source file, runs the one test that should
notice, and fails unless that test reports a failure. Then it restores the file
from git.

It is NOT in CI: each SQL mutation rebuilds the release cdylib, so a full sweep
is minutes rather than seconds. It is a gate you run when you change a test or
the code under one, and its value is that a reviewer can reproduce the proof
with one command instead of believing a paragraph.

    make mutation-check
    scripts/mutation_check.py --list
    scripts/mutation_check.py --only cache_key_drops_the_model

The tree must be clean: mutations are undone with `git checkout --`, which
would take uncommitted work with them.

Stdlib only.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE = "crates/staticembed-core/src"
GLUE = "crates/staticembed-duckdb/src/lib.rs"


@dataclass
class Mutation:
    """One deliberate break, and the test that must notice it."""

    name: str
    file: str
    old: str
    new: str
    #: What must appear as failing. For Rust, the test function name; for SQL,
    #: the substring of the test file name.
    expect_red: str
    #: "rust" or "sql".
    kind: str = "rust"
    #: Extra test names that also legitimately redden; not required, but not
    #: treated as noise either.
    also_reddens: list[str] = field(default_factory=list)


MUTATIONS: list[Mutation] = [
    # ── the bundled model ────────────────────────────────────────────────────
    Mutation(
        name="weights_asset_is_swapped_for_another_file",
        file=f"{CORE}/model.rs",
        old='const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/model.safetensors");',
        new='const WEIGHTS: &[u8] = include_bytes!("../../../models/potion-base-8M/config.json");',
        expect_red="bundled_asset_digests_match_the_pinned_release",
    ),
    Mutation(
        name="embed_always_takes_the_no_token_fallback",
        file=f"{CORE}/model.rs",
        old="        if vector.len() == self.dim {\n            vector\n        } else {",
        new="        if false {\n            vector\n        } else {",
        expect_red="ordinary_text_gets_a_full_width_non_zero_vector",
        also_reddens=["different_strings_get_different_vectors"],
    ),
    Mutation(
        name="embed_returns_a_vector_one_float_short",
        file=f"{CORE}/model.rs",
        old="        let vector = self.inner.encode_single(text);",
        new="        let mut vector = self.inner.encode_single(text);\n        vector.truncate(self.dim - 1);",
        expect_red="every_vector_has_the_declared_width",
    ),
    Mutation(
        name="the_empty_string_is_given_a_one_vector_instead_of_a_zero_vector",
        file=f"{CORE}/model.rs",
        old="        let vector = self.inner.encode_single(text);",
        new="        if text.trim().is_empty() {\n            return vec![1.0_f32; self.dim];\n        }\n        let vector = self.inner.encode_single(text);",
        expect_red="text_with_no_tokens_gets_a_zero_vector_of_full_width",
    ),
    Mutation(
        name="the_model_key_stops_covering_the_weights",
        file=f"{CORE}/model.rs",
        old="        hasher.update(TOKENIZER);\n        hasher.update(WEIGHTS);\n        hasher.update(CONFIG);\n        let key: [u8; 32] = hasher.finalize().into();",
        new="        hasher.update(TOKENIZER);\n        hasher.update(CONFIG);\n        let key: [u8; 32] = hasher.finalize().into();",
        expect_red="the_model_key_covers_the_asset_bytes",
    ),
    Mutation(
        name="embed_becomes_stateful_across_calls",
        file=f"{CORE}/model.rs",
        old="        let vector = self.inner.encode_single(text);",
        new=(
            "        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);\n"
            "        let mut vector = self.inner.encode_single(text);\n"
            "        if CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 0 {\n"
            "            if let Some(first) = vector.first_mut() {\n"
            "                *first += 1.0;\n"
            "            }\n"
            "        }\n"
            "        let vector = vector;"
        ),
        expect_red="embedding_is_deterministic",
    ),
    # ── the cache ────────────────────────────────────────────────────────────
    Mutation(
        name="cache_key_drops_the_model_half",
        file=f"{CORE}/cache.rs",
        old="    hasher.update(CACHE_KEY_DOMAIN);\n    hasher.update(model_key);\n    hasher.update(text.as_bytes());",
        new="    hasher.update(CACHE_KEY_DOMAIN);\n    hasher.update(text.as_bytes());",
        expect_red="the_model_key_is_part_of_the_cache_key",
        also_reddens=["a_cached_vector_does_not_survive_a_model_change"],
    ),
    Mutation(
        name="cache_key_normalises_the_text_the_way_a_tokenizer_might",
        file=f"{CORE}/cache.rs",
        old="    hasher.update(text.as_bytes());",
        new="    hasher.update(text.trim().to_lowercase().as_bytes());",
        expect_red="the_cache_key_uses_the_exact_input_bytes",
    ),
    Mutation(
        name="the_cache_never_rolls_a_generation",
        file=f"{CORE}/cache.rs",
        old="        if self.hot.len() >= self.capacity && !self.hot.contains_key(&key) {",
        new="        if false && self.hot.len() >= self.capacity && !self.hot.contains_key(&key) {",
        expect_red="bounded_cache_never_exceeds_two_generations",
    ),
    Mutation(
        name="a_lookup_never_consults_the_older_generation",
        file=f"{CORE}/cache.rs",
        old="        if let Some(vector) = self.cold.remove(key) {",
        new="        if let Some(vector) = None::<Arc<[f32]>> {",
        expect_red="a_repeatedly_requested_key_is_promoted_out_of_the_older_generation",
    ),
    Mutation(
        name="a_hit_is_not_counted",
        file=f"{CORE}/cache.rs",
        old="        if let Some(vector) = self.hot.get(key) {\n            self.hits += 1;",
        new="        if let Some(vector) = self.hot.get(key) {",
        expect_red="a_hit_returns_the_stored_vector_and_counts_as_a_hit",
    ),
    Mutation(
        name="clearing_leaves_the_counters_where_they_were",
        file=f"{CORE}/cache.rs",
        old="        self.hot.clear();\n        self.cold.clear();\n        self.hits = 0;\n        self.misses = 0;",
        new="        self.hot.clear();\n        self.cold.clear();",
        expect_red="clear_reports_what_it_dropped_and_resets_the_counters",
    ),
    # ── the engine's public behaviour ────────────────────────────────────────
    Mutation(
        name="the_cache_is_never_consulted",
        file=f"{CORE}/lib.rs",
        old="    if let Some(vector) = cache().get(&key) {\n        return Ok(vector);\n    }",
        new="    if let Some(vector) = cache().get(&key) {\n        let _ = vector;\n    }",
        expect_red="repeating_a_value_does_not_re_embed_it",
        also_reddens=["inputs_that_embed_alike_still_occupy_separate_cache_entries"],
    ),
    Mutation(
        name="the_cache_stores_a_zero_vector_instead_of_the_real_one",
        file=f"{CORE}/lib.rs",
        old="    cache().insert(key, Arc::clone(&vector));",
        new="    cache().insert(key, Arc::from(vec![0.0_f32; vector.len()].into_boxed_slice()));",
        expect_red="a_cached_vector_equals_the_uncached_one",
    ),
    Mutation(
        name="clearing_leaves_the_encode_counter_running",
        file=f"{CORE}/lib.rs",
        old="    ENCODED.store(0, Ordering::Relaxed);\n    cache().clear()",
        new="    cache().clear()",
        expect_red="clearing_the_cache_drops_the_entries_and_the_counters",
    ),
    Mutation(
        name="the_version_line_stops_naming_the_model",
        file=f"{CORE}/lib.rs",
        old='            "staticembed {} (model {}@{}, key {}, dim {})",\n            VERSION,\n            model::MODEL_ID,',
        new='            "staticembed {} (model {}@{}, key {}, dim {})",\n            VERSION,\n            "a model",',
        expect_red="describe_names_the_bundled_model_and_the_width",
    ),
    Mutation(
        name="the_pool_becomes_order_sensitive_by_prefixing_the_first_token",
        file=f"{CORE}/model.rs",
        old="        let vector = self.inner.encode_single(text);",
        new='        let vector = self\n            .inner\n            .encode_single(&format!("{} {}", text.split_whitespace().next().unwrap_or(""), text));',
        expect_red="the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not",
        also_reddens=["case_and_surrounding_whitespace_do_not_change_the_vector"],
    ),
    # ── the DuckDB surface ───────────────────────────────────────────────────
    Mutation(
        name="sql_null_stops_propagating_and_becomes_a_zero_vector",
        file=GLUE,
        old="                Cell::Null => vectors.push(None),",
        new="                Cell::Null => vectors.push(Some(staticembed_core::embed(\"\")?)),",
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="a_null_row_is_written_without_its_validity_bit",
        file=GLUE,
        old="                list.set_entry(row, offset, 0);\n                list.set_null(row);",
        new="                list.set_entry(row, offset, 0);",
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="every_row_reads_the_child_vector_from_offset_zero",
        file=GLUE,
        old="                list.set_entry(row, offset, vector.len());\n                offset += vector.len();",
        new="                list.set_entry(row, 0, vector.len());\n                offset += vector.len();",
        expect_red="04_null_and_empty",
        kind="sql",
    ),
    Mutation(
        name="the_scalar_stops_using_the_cache",
        file=GLUE,
        old="Cell::Text(text) => vectors.push(Some(staticembed_core::embed(text)?)),",
        new="Cell::Text(text) => vectors.push(Some(std::sync::Arc::from(\n                    staticembed_core::embed_uncached(text)?.into_boxed_slice(),\n                ))),",
        expect_red="03_a_repeated_query",
        kind="sql",
    ),
    Mutation(
        name="cache_stats_is_folded_to_a_constant_because_it_is_no_longer_volatile",
        file=GLUE,
        old="    /// Without this DuckDB folds a zero-argument scalar to a constant and the\n    /// counters would be read once and reused for the rest of the session.\n    fn volatile() -> bool {\n        true\n    }",
        new="",
        expect_red="01_scalar_composes",
        kind="sql",
        also_reddens=["03_a_repeated_query", "06_text_the_tokenizer"],
    ),
    Mutation(
        name="a_fifth_function_is_registered",
        file=GLUE,
        old='    con.register_scalar_function::<Version>("staticembed_version")?;',
        new='    con.register_scalar_function::<Version>("staticembed_version")?;\n    con.register_scalar_function::<Version>("embed_nearest_neighbour")?;',
        expect_red="05_the_registered_surface",
        kind="sql",
    ),
    Mutation(
        name="embed_returns_a_list_of_doubles_instead_of_floats",
        file=GLUE,
        old="            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),",
        new="            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Double)),",
        expect_red="01_scalar_composes",
        kind="sql",
    ),
]


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def tree_is_clean() -> bool:
    return run(["git", "status", "--porcelain"]).stdout.strip() == ""


def apply(mutation: Mutation) -> None:
    path = REPO_ROOT / mutation.file
    text = path.read_text()
    if mutation.old not in text:
        raise SystemExit(
            f"mutation {mutation.name!r} no longer applies: its anchor is not in {mutation.file}.\n"
            "The code moved; update the mutation rather than deleting it."
        )
    if text.count(mutation.old) != 1:
        raise SystemExit(
            f"mutation {mutation.name!r} anchor appears {text.count(mutation.old)} times in "
            f"{mutation.file}; it must be unique."
        )
    path.write_text(text.replace(mutation.old, mutation.new))


def restore(mutation: Mutation) -> None:
    run(["git", "checkout", "--", mutation.file])


#: `test result: ok. 3 passed; 0 failed; ...`
TEST_RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed;")


class RanNothing(Exception):
    """The command under a mutation executed no test at all.

    This is neither a kill nor a survival, and counting it as either is how a
    mutation harness comes to report a clean sweep having checked nothing. It
    happened here on the first run: `cargo test NAME -- --exact` matches no test,
    because the names are module-qualified, so every Rust mutation "survived"
    against 0 tests run.
    """


def rust_test_failed(test_name: str) -> tuple[bool, str]:
    # No `--exact`: the tests live in a `tests` module, so the bare function
    # name is a substring of the full path rather than equal to it.
    completed = run(["cargo", "test", "--workspace", test_name, "--", "--nocapture"])
    output = completed.stdout + completed.stderr

    executed = sum(int(passed) + int(failed) for passed, failed in TEST_RESULT.findall(output))
    if executed == 0 and "error" not in output:
        raise RanNothing(f"`cargo test {test_name}` matched no test")

    if completed.returncode == 0:
        return False, output
    return f"{test_name} ... FAILED" in output or "test result: FAILED" in output, output


def sql_test_failed(name_fragment: str, duckdb: str) -> tuple[bool, str]:
    built = run(["make", "extension"])
    if built.returncode != 0:
        # A mutation that does not compile is not evidence about a test.
        raise RanNothing(f"the mutated tree did not build:\n{built.stdout}{built.stderr}")
    completed = run(
        [
            sys.executable,
            "scripts/run_sql_tests.py",
            "--extension",
            "build/staticembed.duckdb_extension",
            "--duckdb",
            duckdb,
            "--only",
            name_fragment,
        ]
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 2:
        raise RanNothing(f"--only {name_fragment!r} matched no SQL test file")
    if "PASS " not in output and "FAIL " not in output:
        raise RanNothing(f"the SQL runner reported no result for {name_fragment!r}")
    return completed.returncode == 1 and "FAIL " in output, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="run only mutations whose name contains this substring")
    parser.add_argument("--list", action="store_true", help="print the mutation table and stop")
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        for mutation in MUTATIONS:
            print(f"{mutation.kind:4}  {mutation.name}\n      -> {mutation.expect_red}")
        return 0

    if not tree_is_clean():
        print(
            "the working tree is dirty; mutations are undone with `git checkout --` "
            "and would take uncommitted work with them",
            file=sys.stderr,
        )
        return 2

    selected = [m for m in MUTATIONS if not args.only or args.only in m.name]
    if not selected:
        print(f"no mutation matches {args.only!r}", file=sys.stderr)
        return 2

    survivors = []
    for mutation in selected:
        apply(mutation)
        try:
            if mutation.kind == "rust":
                reddened, output = rust_test_failed(mutation.expect_red)
            else:
                reddened, output = sql_test_failed(mutation.expect_red, args.duckdb)
        except RanNothing as ran_nothing:
            restore(mutation)
            print(f"BROKEN   {mutation.name}: {ran_nothing}", file=sys.stderr)
            return 2
        finally:
            restore(mutation)

        verdict = "KILLED " if reddened else "SURVIVED"
        print(f"{verdict}  {mutation.name}  ->  {mutation.expect_red}")
        if args.verbose or not reddened:
            print(output)
        if not reddened:
            survivors.append(mutation.name)

    print(f"\n{len(selected) - len(survivors)}/{len(selected)} mutations killed")
    if survivors:
        print("surviving mutations — the named test does not detect them:", file=sys.stderr)
        for name in survivors:
            print(f"  {name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
