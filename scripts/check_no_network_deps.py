#!/usr/bin/env python3
"""Fail if anything that can open a socket is linked into the extension.

AC2 says the model loads with no network. A test that calls `embed()` and gets a
vector cannot prove that: it proves the model was found, not that no other code
path could have gone looking for it. What proves it is the absence of an HTTP or
TLS client from the tree at all, and that is mechanical.

Two checks, and they answer different questions.

RUNTIME DEPENDENCY TREE
    `cargo tree --workspace --edges normal` is the set of crates compiled into
    the artifact. Build-dependencies are deliberately excluded: `libduckdb-sys`
    takes `ureq` and `rustls` as build-dependencies to fetch DuckDB's headers,
    and a build-dependency compiles into the build script, not into the cdylib.
    Excluding them is the difference between reporting a real defect and
    reporting the DuckDB bindings.

LINKED LIBRARIES
    The dynamic libraries the built artifact actually names. A vendored OpenSSL
    would not appear as a crate in a stripped tree but would appear here.
    Skipped, loudly, when neither `otool` nor `ldd` is available.

Stdlib only.

    scripts/check_no_network_deps.py
    scripts/check_no_network_deps.py --artifact build/staticembed.duckdb_extension
    scripts/check_no_network_deps.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Crates that speak HTTP, terminate TLS, or exist to fetch a model. Matched
# against the crate name a `cargo tree` line carries, so `rustls-pki-types`
# matches on `rustls`.
FORBIDDEN_CRATES = (
    "openssl",
    "openssl-sys",
    "openssl-src",
    "native-tls",
    "rustls",
    "webpki",
    "ring",
    "ureq",
    "reqwest",
    "hyper",
    "isahc",
    "attohttpc",
    "curl",
    "curl-sys",
    "hf-hub",
    "tokio",
)

# Dynamic library names that mean a TLS or HTTP stack was linked.
FORBIDDEN_LIBRARY_PATTERN = re.compile(r"lib(ssl|crypto|curl|nghttp2)", re.IGNORECASE)

# `name vX.Y.Z` is how every cargo tree line names a crate.
CRATE_LINE = re.compile(r"([A-Za-z0-9_.-]+) v\d")


def crates_in(tree_output: str) -> set[str]:
    return {match.group(1) for match in CRATE_LINE.finditer(tree_output)}


def offending_crates(tree_output: str) -> set[str]:
    return crates_in(tree_output) & set(FORBIDDEN_CRATES)


def runtime_tree() -> str:
    completed = subprocess.run(
        ["cargo", "tree", "--workspace", "--edges", "normal"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"cargo tree failed:\n{completed.stdout}{completed.stderr}")
    return completed.stdout


def linked_libraries(artifact: pathlib.Path) -> list[str] | None:
    if shutil.which("otool"):
        command = ["otool", "-L", str(artifact)]
    elif shutil.which("ldd"):
        command = ["ldd", str(artifact)]
    else:
        return None
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{command[0]} failed:\n{completed.stdout}{completed.stderr}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def self_test() -> int:
    """Prove the matcher detects what it is looking for.

    A checker that returns clean on a clean tree and has never been shown a
    dirty one is indistinguishable from a checker that always returns clean.
    """
    dirty = (
        "staticembed-core v0.1.0\n"
        "├── model2vec-rs v0.2.1\n"
        "│   ├── hf-hub v0.4.3\n"
        "│   │   └── ureq v2.12.1\n"
        "│   │       └── rustls v0.23.43\n"
        "└── sha2 v0.10.9\n"
    )
    found = offending_crates(dirty)
    expected = {"hf-hub", "ureq", "rustls"}
    if found != expected:
        print(f"self-test FAILED: matched {sorted(found)}, expected {sorted(expected)}", file=sys.stderr)
        return 1

    clean = "staticembed-core v0.1.0\n└── sha2 v0.10.9\n"
    if offending_crates(clean):
        print("self-test FAILED: matched a clean tree", file=sys.stderr)
        return 1

    if not FORBIDDEN_LIBRARY_PATTERN.search("/usr/lib/libssl.3.dylib"):
        print("self-test FAILED: the library pattern misses libssl", file=sys.stderr)
        return 1
    if FORBIDDEN_LIBRARY_PATTERN.search("/usr/lib/libSystem.B.dylib"):
        print("self-test FAILED: the library pattern matches libSystem", file=sys.stderr)
        return 1

    print("self-test ok: the matcher finds a planted TLS stack and clears a clean tree")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    failures = 0

    found = offending_crates(runtime_tree())
    if found:
        print(
            "FAIL: the runtime dependency tree contains "
            + ", ".join(sorted(found))
            + "\n      The extension bundles its model; nothing in it should be able to fetch one.",
            file=sys.stderr,
        )
        failures += 1
    else:
        print(f"ok: no HTTP or TLS crate in the runtime dependency tree ({len(FORBIDDEN_CRATES)} names checked)")

    if args.artifact:
        if not args.artifact.is_file():
            print(f"FAIL: no artifact at {args.artifact}", file=sys.stderr)
            return 1
        libraries = linked_libraries(args.artifact)
        if libraries is None:
            print("SKIPPED: neither otool nor ldd is available, so linked libraries were not checked")
        else:
            offending = [line for line in libraries if FORBIDDEN_LIBRARY_PATTERN.search(line)]
            if offending:
                print("FAIL: the artifact links a TLS or HTTP library:", file=sys.stderr)
                for line in offending:
                    print(f"       {line}", file=sys.stderr)
                failures += 1
            else:
                print(f"ok: the artifact links no TLS or HTTP library ({len(libraries)} entries checked)")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
