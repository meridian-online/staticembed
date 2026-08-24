#!/usr/bin/env python3
"""Fail when the documented SQL surface disagrees with the loaded catalog.

`README.md` and the `staticembed-duckdb` module doc both write out the
extension's function table, including the field list of the STRUCT that
`staticembed_cache_stats()` returns. Three copies of one signature is three
chances to be wrong, and one of them was: the page listed five fields for a
six-field struct, omitting `uncached` — the counter that tells a reader which
side of the cache's bound they are on.

DuckDB settles it mechanically. `duckdb_functions()` reports what a loaded
extension really registered, with the return type and, for a STRUCT, its field
names in order. This loads the local build and compares.

WHAT IS DERIVED (the source of truth)
    `duckdb_functions()` after `LOAD`, taken as the delta against a snapshot
    made before the load, so nothing built into the CLI is mistaken for ours.
    `extension_directory` points at an empty temporary directory, so an
    installed community build cannot be picked up instead.

WHAT IS ASSERTED
    1. Every `STRUCT(...)` signature written in a scanned file names exactly the
       catalog's fields, in the catalog's order.
    2. Every function the catalog reports appears in README's function table,
       and every row of that table names a function the catalog has.

WHAT IS NOT ASSERTED
    Behaviour. This says the docs describe the registered surface; whether that
    surface does anything is `test/sql/`.

Needs the duckdb CLI and a packaged extension. Stdlib only.

    scripts/check_documented_surface.py --extension build/staticembed.duckdb_extension
    scripts/check_documented_surface.py --self-test
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Files that write the surface out in prose and must agree with the catalog.
SCANNED = ("README.md", "crates/staticembed-duckdb/src/lib.rs")

#: A `STRUCT(...)` signature as the docs write it: field names, no types.
DOCUMENTED_STRUCT = re.compile(r"STRUCT\(([^)]*)\)")

#: A `STRUCT(...)` signature as DuckDB writes it: `name TYPE, name TYPE`.
CATALOG_STRUCT = re.compile(r"^STRUCT\((.*)\)$", re.DOTALL)

#: A function name as README's table writes it, in backticks with parentheses.
DOCUMENTED_FUNCTION = re.compile(r"`(\w+)\(")


def catalog(duckdb: str, extension: pathlib.Path) -> dict[str, str]:
    """Function name to return type, for the functions this extension registers."""
    with tempfile.TemporaryDirectory() as workdir:
        work = pathlib.Path(workdir)
        extensions = work / "extensions"
        extensions.mkdir()
        script = f"""
SET extension_directory='{extensions}';
CREATE TABLE before AS SELECT DISTINCT function_name FROM duckdb_functions();
LOAD '{extension}';
SELECT function_name || chr(9) || return_type
FROM (SELECT DISTINCT function_name, return_type FROM duckdb_functions()
      WHERE function_name NOT IN (SELECT function_name FROM before))
ORDER BY function_name;
"""
        completed = subprocess.run(
            [duckdb, "-unsigned", "-init", os.devnull, "-batch", "-noheader", "-list"],
            input=script,
            text=True,
            capture_output=True,
            cwd=work,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"reading the catalog failed:\n{completed.stdout}{completed.stderr}")

    found = {}
    for line in completed.stdout.splitlines():
        if "\t" not in line:
            continue
        name, return_type = line.split("\t", 1)
        found[name.strip()] = return_type.strip()
    if not found:
        raise SystemExit("the catalog delta is empty; the extension registered nothing")
    return found


def catalog_struct_fields(return_type: str) -> list[str] | None:
    """Field names, in order, from a catalog STRUCT return type."""
    match = CATALOG_STRUCT.match(return_type)
    if not match:
        return None
    return [field.strip().split()[0] for field in match.group(1).split(",") if field.strip()]


def documented_struct_fields(text: str) -> list[list[str]]:
    """Every STRUCT field list a document writes out."""
    signatures = []
    for match in DOCUMENTED_STRUCT.finditer(text):
        fields = [field.strip() for field in match.group(1).split(",") if field.strip()]
        # A signature written with types is the catalog's own spelling quoted
        # back; take the names so both forms compare alike.
        signatures.append([field.split()[0] for field in fields])
    return signatures


def self_test() -> int:
    """Prove each matcher sees what it is looking for."""
    catalog_type = "STRUCT(hits BIGINT, misses BIGINT, encoded BIGINT, uncached BIGINT)"
    fields = catalog_struct_fields(catalog_type)
    if fields != ["hits", "misses", "encoded", "uncached"]:
        print(f"self-test FAILED: catalog parse gave {fields}", file=sys.stderr)
        return 1
    if catalog_struct_fields("BIGINT") is not None:
        print("self-test FAILED: a non-STRUCT was parsed as one", file=sys.stderr)
        return 1

    stale = "| `staticembed_cache_stats()` | `STRUCT(hits, misses, encoded, entries, capacity)` |"
    found = documented_struct_fields(stale)
    if found != [["hits", "misses", "encoded", "entries", "capacity"]]:
        print(f"self-test FAILED: doc parse gave {found}", file=sys.stderr)
        return 1
    if found[0] == fields:
        print("self-test FAILED: the stale signature compared equal", file=sys.stderr)
        return 1

    reordered = documented_struct_fields("STRUCT(misses, hits)")
    if reordered == [["hits", "misses"]]:
        print("self-test FAILED: field order is not being compared", file=sys.stderr)
        return 1

    row = "| `staticembed_cache_stats()` | `STRUCT(hits, misses)` | what it does |"
    match = DOCUMENTED_FUNCTION.search(row)
    if match is None or match.group(1) != "staticembed_cache_stats":
        print("self-test FAILED: the function matcher missed a table row", file=sys.stderr)
        return 1
    if DOCUMENTED_FUNCTION.findall(row)[1:] != ["STRUCT"]:
        print(
            "self-test FAILED: the return type in a row no longer looks like a call, so "
            "taking only the first match is no longer the thing that excludes it",
            file=sys.stderr,
        )
        return 1

    print(
        "self-test ok: the catalog parser reads a STRUCT and rejects a scalar, the doc parser "
        "reads a written signature, a stale field list compares unequal, and field order counts"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=pathlib.Path)
    parser.add_argument("--duckdb", default=os.environ.get("DUCKDB", "duckdb"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.extension is None:
        print("--extension is required unless --self-test is given", file=sys.stderr)
        return 2
    duckdb = shutil.which(args.duckdb)
    if duckdb is None:
        print(f"the duckdb CLI ({args.duckdb}) is not on PATH", file=sys.stderr)
        return 2
    if not args.extension.is_file():
        print(f"no packaged extension at {args.extension}; run `make extension`", file=sys.stderr)
        return 2

    registered = catalog(duckdb, args.extension.resolve())
    struct_fields = {
        name: fields
        for name, return_type in registered.items()
        if (fields := catalog_struct_fields(return_type)) is not None
    }

    failures = 0
    for relative in SCANNED:
        path = REPO_ROOT / relative
        text = path.read_text()
        for written in documented_struct_fields(text):
            if not any(written == fields for fields in struct_fields.values()):
                print(
                    f"FAIL: {relative} writes STRUCT({', '.join(written)}), which matches no "
                    f"registered function. The catalog has: "
                    + "; ".join(
                        f"{name} STRUCT({', '.join(fields)})"
                        for name, fields in struct_fields.items()
                    ),
                    file=sys.stderr,
                )
                failures += 1

    readme = (REPO_ROOT / "README.md").read_text()
    table_rows = [line for line in readme.splitlines() if line.startswith("| `")]
    # The first backticked call in a row is the function; later ones are its
    # return type, and `STRUCT(` looks exactly like a call.
    documented = {
        match.group(1) for row in table_rows if (match := DOCUMENTED_FUNCTION.search(row))
    }
    missing = set(registered) - documented
    invented = documented - set(registered)
    if missing:
        print(f"FAIL: README's table omits registered functions: {sorted(missing)}", file=sys.stderr)
        failures += 1
    if invented:
        print(f"FAIL: README's table names functions the catalog has not: {sorted(invented)}", file=sys.stderr)
        failures += 1

    if failures:
        return 1
    print(
        f"ok: {len(registered)} registered functions, "
        f"{len(struct_fields)} STRUCT signatures, and every signature written in "
        f"{len(SCANNED)} files matches the loaded catalog"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
