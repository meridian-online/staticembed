#!/usr/bin/env python3
"""Run every SQL example in `description.yml`, and check the entry describes this tree.

`description.yml` is the registry submission. What it says is what a stranger
reads on the extension's page and pastes into a shell, and **nothing upstream
ever runs it**: the `doc_test` job in `duckdb/community-extensions`'
`.github/workflows/build.yml` carries `if: false`, so it is not merely skipped
for a given run, it is switched off. On a real registry PR the observed result
is 7 pass / 6 skip / 0 fail, with `doc_test` among the skips. A published
example that throws on paste therefore stays published until a person complains.

WHAT THIS RUNS. Every example, in ONE DuckDB session, in the order the page
presents them, against a packaged artifact. The session is given nothing the
descriptor does not create for itself: the prelude is an `extension_directory`
pointed at an empty temporary directory and a `LOAD` of the artifact, and that
is all. A later example may use a table an earlier one created, because that is
how someone reads a page; it may not use one nobody created.

The environment is scrubbed the same way `run_sql_tests.py` scrubs it — `HOME`
and the model-cache variables point at empty temporary directories and the proxy
variables are removed — so an example that only works because the developer's
machine has something in it fails here.

HOW THE EXAMPLES ARE FOUND. With a YAML parser, not a regex over the file.
`hello_world` and `extended_description` are block scalars, and their content is
prose that contains lines beginning `extension:`, `repo:` and `## ` — a
line-oriented pattern reading this file as text stops at the first of those and
silently drops the rest of the examples, which is a checker that passes because
it found nothing. Inside `extended_description`, examples are the ```sql fenced
blocks; a ```python or ```text block is not one.

WHAT ELSE IT CHECKS. The entry has to describe THIS tree, and the parts that can
go stale silently are the ones asserted here: the version against `Cargo.toml`,
the repository against the `repository` field in `Cargo.toml`, the platform
exclusions against the distribution matrix that defines those names, and the
function table against `duckdb_functions()` of the loaded artifact — derived
from the build rather than from a list written here, so registering a sixth
function reddens this until the page mentions it.

Needs PyYAML. The registry's own `scripts/build.py` does too.

    scripts/check_description_examples.py --extension build/staticembed.duckdb_extension
    scripts/check_description_examples.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DESCRIPTION = REPO_ROOT / "description.yml"
MATRIX = REPO_ROOT / "extension-ci-tools" / "config" / "distribution_matrix.json"

MARKER = "STATICEMBED_EXAMPLE"

PRELUDE = """\
SET extension_directory='{extension_directory}';
LOAD '{extension}';
"""

SCRUBBED_ENV_TO_TEMP = ("HOME", "XDG_CACHE_HOME", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")
SCRUBBED_ENV_TO_UNSET = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")


def load_yaml(path: pathlib.Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "PyYAML is needed to read description.yml with a parser rather than a regex.\n"
            "    pip install pyyaml"
        )
    return yaml.safe_load(path.read_text())


def fenced_sql(markdown: str) -> list[str]:
    """Every ```sql block, in order.

    A fence is a line whose first non-space characters are three or more
    backticks. The opening fence's info string decides whether the block counts;
    the closing fence is the next fence line of at least the same length. Blocks
    tagged anything other than `sql` are skipped rather than run, which is what
    keeps a ```text sample of output from being executed as a query.
    """
    blocks: list[str] = []
    inside = False
    is_sql = False
    collected: list[str] = []
    fence_length = 0
    for line in markdown.splitlines():
        match = re.match(r"(`{3,})\s*([A-Za-z0-9_+-]*)\s*$", line.lstrip())
        if not inside:
            if match:
                inside = True
                fence_length = len(match.group(1))
                is_sql = match.group(2).lower() == "sql"
                collected = []
            continue
        if match and len(match.group(1)) >= fence_length:
            if is_sql:
                blocks.append("\n".join(collected))
            inside = False
            continue
        collected.append(line)
    return blocks


def examples(descriptor: dict) -> list[tuple[str, str]]:
    """(label, sql) for every example the entry publishes."""
    docs = descriptor.get("docs") or {}
    found: list[tuple[str, str]] = []
    hello = docs.get("hello_world")
    if hello:
        found.append(("docs.hello_world", hello))
    extended = docs.get("extended_description")
    if extended:
        for index, block in enumerate(fenced_sql(extended), start=1):
            found.append((f"docs.extended_description sql block {index}", block))
    return found


def scrubbed_environment(sandbox: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in SCRUBBED_ENV_TO_TEMP:
        env[name] = str(sandbox)
    for name in SCRUBBED_ENV_TO_UNSET:
        env.pop(name, None)
    return env


def run_session(
    duckdb: str,
    extension: pathlib.Path | None,
    blocks: list[tuple[str, str]],
) -> tuple[bool, str, int]:
    """Run every block in one session. Returns (ok, output, blocks_reached)."""
    script = []
    for index, (_label, sql) in enumerate(blocks, start=1):
        script.append(f"SELECT '{MARKER}_{index}' AS example;")
        script.append(sql)
    body = "\n".join(script)

    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extension_directory = work / "extensions"
        extension_directory.mkdir()
        sandbox = work / "home"
        sandbox.mkdir()

        prelude = (
            PRELUDE.format(extension_directory=extension_directory, extension=extension)
            if extension is not None
            else f"SET extension_directory='{extension_directory}';\n"
        )
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-box"],
            input=prelude + body,
            text=True,
            capture_output=True,
            cwd=work,
            env=scrubbed_environment(sandbox),
        )

    output = completed.stdout + completed.stderr
    reached = len(re.findall(rf"{MARKER}_(\d+)", output))
    ok = completed.returncode == 0 and "Error:" not in output
    return ok, output, reached


def registered_functions(duckdb: str, extension: pathlib.Path) -> list[str]:
    """The function names the artifact adds to the catalog, derived by loading it."""
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extension_directory = work / "extensions"
        extension_directory.mkdir()
        sandbox = work / "home"
        sandbox.mkdir()
        script = (
            f"SET extension_directory='{extension_directory}';\n"
            "CREATE TABLE before AS SELECT DISTINCT function_name FROM duckdb_functions();\n"
            f"LOAD '{extension}';\n"
            "SELECT function_name FROM (SELECT DISTINCT function_name FROM duckdb_functions()) "
            "WHERE function_name NOT IN (SELECT function_name FROM before) ORDER BY 1;\n"
        )
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-noheader", "-csv"],
            input=script,
            text=True,
            capture_output=True,
            cwd=work,
            env=scrubbed_environment(sandbox),
        )
    if completed.returncode != 0:
        raise SystemExit(f"could not read the catalog from {extension}:\n{completed.stdout}{completed.stderr}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def repository_in_cargo_toml() -> str:
    """`owner/name` from the `repository` URL in Cargo.toml."""
    text = (REPO_ROOT / "Cargo.toml").read_text()
    found = re.search(r'^\s*repository\s*=\s*"https://github\.com/([^"/]+/[^"/]+?)/?"', text, re.MULTILINE)
    if not found:
        raise SystemExit("Cargo.toml has no github repository URL")
    return found.group(1)


def known_platforms() -> set[str] | None:
    """Every `duckdb_arch` the distribution matrix defines, or None if absent."""
    if not MATRIX.is_file():
        return None
    matrix = json.loads(MATRIX.read_text())
    return {
        entry["duckdb_arch"]
        for group in matrix.values()
        for entry in group.get("include", [])
    }


def descriptor_problems(descriptor: dict, functions: list[str] | None) -> list[str]:
    """Everything about the entry that disagrees with this tree."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_artifact_version import version_in_cargo_toml

    problems: list[str] = []
    extension = descriptor.get("extension") or {}
    repo = descriptor.get("repo") or {}

    expected_version = version_in_cargo_toml()
    if str(extension.get("version")) != expected_version:
        problems.append(
            f"extension.version is {extension.get('version')!r}, but Cargo.toml says {expected_version!r}"
        )

    expected_repo = repository_in_cargo_toml()
    if extension.get("name") != expected_repo.split("/")[-1]:
        problems.append(
            f"extension.name is {extension.get('name')!r}, but the repository is {expected_repo!r} — "
            f"the registry requires the directory, the name and the built library to agree"
        )
    if repo.get("github") != expected_repo:
        problems.append(f"repo.github is {repo.get('github')!r}, but Cargo.toml says {expected_repo!r}")

    ref = str(repo.get("ref", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        problems.append(
            f"repo.ref is {ref!r}; the registry pins a full 40-character commit SHA, and an "
            f"abbreviated one or a branch name is not a pin"
        )

    platforms = known_platforms()
    if platforms is not None:
        for field in ("excluded_platforms", "opt_in_platforms"):
            for name in filter(None, str(extension.get(field, "")).split(";")):
                if name not in platforms:
                    problems.append(
                        f"extension.{field} names {name!r}, which is not a duckdb_arch in the "
                        f"distribution matrix — a misspelt exclusion excludes nothing"
                    )

    if functions is not None:
        docs = descriptor.get("docs") or {}
        page = f"{docs.get('hello_world', '')}\n{docs.get('extended_description', '')}"
        for name in functions:
            if not re.search(rf"\b{re.escape(name)}\b", page):
                problems.append(
                    f"the artifact registers {name!r} and the entry never names it — "
                    f"the page a stranger reads would not mention a function they have"
                )

    return problems


def self_test() -> int:
    """Prove the extractor extracts and the runner reddens.

    Every case plants a specific defect and requires it to be reported. A
    checker that clears the file it was written against and has never been shown
    a broken one is indistinguishable from a checker that always clears.
    """
    failures: list[str] = []

    def expect(label: str, condition: bool) -> None:
        if not condition:
            failures.append(label)

    # A block scalar whose CONTENT contains lines that look like top-level YAML
    # keys and a markdown heading. This is the case a line-oriented regex over
    # the file gets wrong: it ends the block at `repo:` and finds one example
    # where there are three.
    synthetic = """\
extension:
  name: staticembed
  version: 9.9.9
repo:
  github: example/staticembed
  ref: deadbeef
docs:
  hello_world: |
    SELECT embed('hello') AS hello;
    -- repo: this line is prose, not a key
    SELECT 2 AS also_hello;
  extended_description: |
    Prose about the extension.

    ```sql
    SELECT 3 AS first_block;
    ```

    ## extension: a heading that looks like a key

    ```python
    print("not sql, and must not be run")
    ```

    ```text
    extension: neither is this
    ```

    ```sql
    SELECT 4 AS second_block;
    ```
"""
    with tempfile.TemporaryDirectory() as workdir:
        path = pathlib.Path(workdir) / "description.yml"
        path.write_text(synthetic)
        descriptor = load_yaml(path)

        found = examples(descriptor)
        expect("finds the hello_world block and both sql blocks", len(found) == 3)
        if len(found) == 3:
            expect("hello_world survives a line that looks like a key", "also_hello" in found[0][1])
            expect("the first sql block is the first sql block", "first_block" in found[1][1])
            expect("the second sql block is found past a python block", "second_block" in found[2][1])
        expect(
            "the python block is not extracted",
            all("not sql" not in sql for _label, sql in found),
        )
        expect(
            "the text block is not extracted",
            all("neither is this" not in sql for _label, sql in found),
        )

        # Descriptor agreement: each of these must be reported.
        problems = descriptor_problems(descriptor, ["embed", "a_function_the_page_never_names"])
        expect("a wrong version is caught", any("extension.version" in p for p in problems))
        expect("a wrong repository is caught", any("repo.github" in p for p in problems))
        expect("an abbreviated ref is caught", any("repo.ref" in p for p in problems))
        expect(
            "a registered function the page never names is caught",
            any("a_function_the_page_never_names" in p for p in problems),
        )
        expect(
            "a function the page does name is not reported",
            not any("'embed'" in p for p in problems),
        )

    # A misspelt platform exclusion, checked only when the matrix is available.
    if known_platforms() is not None:
        typo = {
            "extension": {
                "name": "staticembed",
                "version": "0.0.0",
                "excluded_platforms": "wasm_mvp;windows_amd64_mingww",
            },
            "repo": {"github": "x/y", "ref": "0" * 40},
        }
        expect(
            "a misspelt platform exclusion is caught",
            any("windows_amd64_mingww" in p for p in descriptor_problems(typo, None)),
        )
        expect(
            "a real platform name is not reported",
            not any("'wasm_mvp'" in p for p in descriptor_problems(typo, None)),
        )
    else:
        print("self-test: SKIPPED the platform-name check — extension-ci-tools is not checked out")

    # And the runner itself. No extension needed: this proves that a failing
    # example makes this script fail, which is the whole point of running them.
    duckdb = shutil.which("duckdb")
    if duckdb is None:
        print("self-test: SKIPPED the runner check — the duckdb CLI is not on PATH")
    else:
        ok, _output, reached = run_session(duckdb, None, [("good", "SELECT 1 AS a;")])
        expect("a working example passes", ok and reached == 1)

        ok, output, reached = run_session(
            duckdb,
            None,
            [("first", "SELECT 1 AS a;"), ("second", "SELECT * FROM a_table_nobody_created;")],
        )
        expect("an example referring to a fixture nobody created fails", not ok)
        expect("and the output says how far the session got", reached == 2)
        expect("and names the missing table", "a_table_nobody_created" in output)

        ok, _output, _reached = run_session(duckdb, None, [("syntax", "SELEKT 1;")])
        expect("a syntactically broken example fails", not ok)

    if failures:
        for failure in failures:
            print(f"self-test FAILED: {failure}", file=sys.stderr)
        return 1

    print(
        "self-test ok: the extractor finds a block scalar past lines that look like YAML keys, "
        "takes sql fences and leaves python and text ones, and the runner fails on a missing "
        "fixture and on a syntax error"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extension", type=pathlib.Path)
    parser.add_argument("--description", type=pathlib.Path, default=DESCRIPTION)
    parser.add_argument("--duckdb", default="duckdb")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.extension is None:
        parser.error("--extension is required: the examples are run against a packaged artifact")
    if not args.extension.is_file():
        raise SystemExit(f"no artifact at {args.extension}")
    # The session runs with its cwd in a temporary directory, so a relative path
    # would resolve to nothing there.
    args.extension = args.extension.resolve()
    if shutil.which(args.duckdb) is None:
        raise SystemExit(f"{args.duckdb} is not on PATH; the examples are run through the DuckDB CLI")

    descriptor = load_yaml(args.description)
    failures = 0

    functions = registered_functions(args.duckdb, args.extension)
    problems = descriptor_problems(descriptor, functions)
    if problems:
        print(f"FAIL: {args.description} does not describe this tree:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        failures += 1
    else:
        print(
            f"ok: the entry's version, repository, ref shape and platform names agree with the "
            f"tree, and it names all {len(functions)} functions the artifact registers"
        )

    blocks = examples(descriptor)
    if not blocks:
        print(f"FAIL: {args.description} publishes no SQL example", file=sys.stderr)
        return 1

    ok, output, reached = run_session(args.duckdb, args.extension, blocks)
    if not ok:
        stopped = blocks[min(reached, len(blocks)) - 1][0] if reached else "the prelude"
        print(
            f"FAIL: the published examples do not run. The session stopped in {stopped} "
            f"({reached} of {len(blocks)} reached).",
            file=sys.stderr,
        )
        print(output, file=sys.stderr)
        failures += 1
    else:
        print(
            f"ok: all {len(blocks)} published examples ran in one session against "
            f"{args.extension}, with no fixture the entry does not create for itself"
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
