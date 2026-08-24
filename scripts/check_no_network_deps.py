#!/usr/bin/env python3
"""Fail if anything that can open a socket is linked into the extension.

AC2 says the model loads with no network. A test that calls `embed()` and gets a
vector cannot prove that: it proves the model was found, not that no other code
path could have gone looking for it. What proves it is the absence of an HTTP or
TLS client from the tree at all, and that is mechanical.

Two checks, and they answer different questions.

UNDEFINED SYMBOLS
    Every call a binary makes into the platform appears in its undefined symbol
    table. So if none of the socket API's entry points is undefined, that binary
    makes no direct socket call. This check came from the reviewer of the first
    version of this file, who found the assertion it was missing.

    On its own it is not a proof of absence, and it is worth knowing why:
    `/usr/bin/curl` passes it. curl opens sockets through `libcurl.4.dylib`, so
    the socket calls are undefined in the library rather than in the binary. It
    is the LINKED LIBRARIES check below that fails curl.

    Together the two are a proof of absence for this artifact, and the argument
    is short enough to state. It links four things: itself, libiconv,
    CoreFoundation and libSystem. libiconv converts charsets. Every route to a
    socket that the other two offer — the BSD calls in libSystem, and CFSocket
    and CFStream in CoreFoundation — is in the list below, and none of them is
    undefined in the artifact. There is nowhere else to delegate to.

    What neither covers: raw syscall instructions that bypass libSystem, and a
    `dlopen` of a library named at runtime. Nothing in this tree does either.

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

# Entry points to the platform's socket API. A binary that can reach the network
# calls at least one of these, so their joint absence from the undefined symbol
# table is the thing that makes "no network" a proof rather than a hope.
#
# BSD sockets, name resolution, and the two macOS frameworks that offer a
# connection without going through the BSD calls.
FORBIDDEN_SYMBOLS = (
    "socket",
    "socketpair",
    "connect",
    "connectx",
    "bind",
    "listen",
    "accept",
    "accept4",
    "send",
    "sendto",
    "sendmsg",
    "sendmmsg",
    "recv",
    "recvfrom",
    "recvmsg",
    "recvmmsg",
    "shutdown",
    "setsockopt",
    "getsockopt",
    "getpeername",
    "getsockname",
    "getaddrinfo",
    "freeaddrinfo",
    "getnameinfo",
    "gethostbyname",
    "gethostbyname2",
    "gethostbyaddr",
    "res_query",
    "res_search",
    "res_init",
    "CFSocketCreate",
    "CFStreamCreatePairWithSocketToHost",
    "CFReadStreamOpen",
    "SSLHandshake",
    "SSLCreateContext",
    "nw_connection_create",
    "nw_connection_start",
    "nw_endpoint_create_host",
)

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


def parse_nm_output(text: str) -> list[str]:
    """Bare symbol names from `nm` output.

    Leading underscores (Mach-O prefixes every C symbol with one) and
    `@GLIBC_2.x` version suffixes (ELF) are stripped, so a name in
    FORBIDDEN_SYMBOLS matches on either platform.

    Split out from the `nm` call so that `--self-test` can drive it with real
    `nm` output. The version of the self-test this replaces normalised its own
    planted symbol and then matched that, which tested the assertion rather than
    the parser: deleting the `.lstrip("_")` here left the self-test exiting 0
    and still printing that it caught all 38 names, while the macOS check
    cleared `/usr/bin/nc`.
    """
    names = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        names.append(fields[-1].lstrip("_").split("@", 1)[0])
    return names


def undefined_symbols(artifact: pathlib.Path) -> list[str] | None:
    """The artifact's undefined symbols, one bare name per entry."""
    if not shutil.which("nm"):
        return None
    for arguments in (["-u"], ["-D", "-u"], ["--dynamic", "--undefined-only"]):
        completed = subprocess.run(
            ["nm", *arguments, str(artifact)], text=True, capture_output=True, check=False
        )
        if completed.returncode == 0 and completed.stdout.strip():
            break
    else:
        raise SystemExit(f"nm reported no undefined symbols for {artifact}; that is not credible")

    return parse_nm_output(completed.stdout)


def offending_symbols(names: list[str]) -> set[str]:
    """Which forbidden entry points appear, matched whole rather than as
    substrings.

    Substring matching is what made the first draft of this useless: `bind`
    matched `dyld_stub_binder`, which is the dynamic linker and not a socket.
    """
    return set(names) & set(FORBIDDEN_SYMBOLS)


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

    # Every forbidden symbol, one at a time, in both platforms' spellings, as
    # lines of real `nm` output driven through the real parser. Planting a
    # pre-normalised name here instead is what made the previous version of this
    # unable to notice its own parser going blind.
    for symbol in FORBIDDEN_SYMBOLS:
        planted = [
            f"                 U _{symbol}",  # macOS `nm -u`
            f"                 U {symbol}",  # ELF `nm -D -u`
            f"                 U {symbol}@GLIBC_2.2.5",  # ELF, versioned
            f"_{symbol}",  # macOS `nm -u`, bare-name form
        ]
        for line in planted:
            caught = offending_symbols(parse_nm_output(line))
            if caught != {symbol}:
                print(
                    f"self-test FAILED: {line!r} parsed to {sorted(caught)}, expected [{symbol!r}]",
                    file=sys.stderr,
                )
                return 1

    # Real `nm -u` output from a clean build of this extension. Every one of
    # these must pass, or the check would fail on its own artifact.
    clean_output = """\
                 U dyld_stub_binder
                 U _malloc
                 U _free
                 U _open
                 U _read
                 U _write
                 U _close
                 U _pthread_create
                 U _pthread_cond_wait
                 U _getentropy
                 U _sysconf
                 U _mmap
                 U _sched_yield
"""
    caught = offending_symbols(parse_nm_output(clean_output))
    if caught:
        print(f"self-test FAILED: clean symbols matched {sorted(caught)}", file=sys.stderr)
        return 1

    # And end to end, against a binary on this machine that really does open
    # sockets. Synthetic input proves the parser; this proves the whole path
    # from `nm` through to the verdict.
    networking = next(
        (
            candidate
            for candidate in (
                pathlib.Path("/usr/bin/nc"),
                pathlib.Path("/bin/nc"),
                pathlib.Path("/usr/bin/ping"),
                pathlib.Path("/bin/ping"),
                pathlib.Path("/usr/bin/ssh"),
            )
            if candidate.is_file()
        ),
        None,
    )
    if networking is None:
        print("self-test: SKIPPED the live check — no networking binary found to test against")
    else:
        live = undefined_symbols(networking)
        if live is None:
            print("self-test: SKIPPED the live check — nm is not available")
        elif not offending_symbols(live):
            print(
                f"self-test FAILED: {networking} opens sockets and the check cleared it",
                file=sys.stderr,
            )
            return 1
        else:
            print(
                f"self-test ok: {networking} is correctly flagged "
                f"({len(offending_symbols(live))} socket entry points found in it)"
            )

    print(
        f"self-test ok: the matcher finds a planted TLS stack, catches all "
        f"{len(FORBIDDEN_SYMBOLS)} socket entry points parsed from real nm output in "
        f"both platforms' spellings, and clears a clean tree and a clean symbol table"
    )
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

        symbols = undefined_symbols(args.artifact)
        if symbols is None:
            print("SKIPPED: nm is not available, so the undefined symbols were not checked")
        else:
            caught = offending_symbols(symbols)
            if caught:
                print(
                    "FAIL: the artifact can reach the network — its undefined symbols include "
                    + ", ".join(sorted(caught)),
                    file=sys.stderr,
                )
                failures += 1
            else:
                print(
                    f"ok: none of the {len(FORBIDDEN_SYMBOLS)} socket entry points is undefined "
                    f"in the artifact ({len(symbols)} undefined symbols in total), so it makes "
                    f"no direct socket call"
                )

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
