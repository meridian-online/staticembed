#!/usr/bin/env python3
"""Neuter one branch of a checked script at a time, and require the script to notice.

A test that passes against working code and against deliberately broken code is
testing nothing. `scripts/mutation_check.py` says that about the product and
proves it for a hand-written table of breakages. This says it about the checks
themselves, and enumerates the breakages from the source rather than from a
table, because the table is written by the same person as the check and misses
the same things they did.

Every review of `check_quality_claims.py` so far has found an assertion that
could be deleted with every gate green, and every one of those was found by a
person reading the file. Reading it once more would have found the next. Running
it does not depend on anybody's attention, and the first thing it did when
pointed at itself was report two site kinds its own self-test never exercised:

WHAT IT DOES
    For each `Target` below, parse the script and enumerate every site where a
    decision is taken — `if <test>:`, `for <name> in <iterable>:`, and
    `problems += <call>` — outside the functions named in `not_swept`, which are
    the script's own self-test machinery and would only be measuring itself.
    Neuter one site at a time (`if False:`, `for x in []:`, `problems += []`),
    run the script's own commands, and require at least one of them to come back
    non-zero. A site where every command still exits 0 is a line that can be
    deleted with the gate reporting itself green.

    A site that stays green must be written into `allowed` with the reason it
    cannot fail, and an entry there that no longer names a live green site is
    reported too — a permission that outlives what it permitted quietly widens
    what the sweep tolerates. That is the same shape as `ALLOWED_UNIVERSALS` in
    `check_quality_claims.py`, and for the same reason.

WHAT IT DOES NOT DO
    It does not neuter a guard in the other direction. `if <test>:` becomes
    `if False:` and never `if True:`, so a condition that must *not* fire — a
    fast path, an early return — is not measured here.

    It does not touch expressions. A comparison flipped from `!=` to `==`, a
    regex that matches nothing, a constant edited: none of those are branches
    and none are enumerated. `SENTENCE_END` compiling to something that matches
    nothing was one of those, and it is a real hole this cannot see. The
    hand-written table in `scripts/mutation_check.py` is where a breakage of
    that shape goes.

    It does not say the assertions are the right ones. A check that asserts
    nothing useful, thoroughly, passes this.

    It sweeps the scripts in `TARGETS` and no others. Adding one is a `Target`
    entry plus however many `allowed` entries its first run turns up, and each
    of those is a sentence somebody has to mean.

Stdlib only.

    scripts/check_assertions_can_fail.py
    scripts/check_assertions_can_fail.py --self-test
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Long enough that a slow machine is not a failure, short enough that a neuter
#: which turns a loop infinite does not hold a CI runner for an hour.
TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class Allowed:
    """A site that stays green when neutered, and why it cannot be otherwise."""

    #: The function the site is in, innermost first.
    function: str
    #: The exact source text of the expression that gets neutered, so that an
    #: entry stops matching when the line it permits is edited.
    source: str
    why: str


@dataclass(frozen=True)
class Target:
    """A script to sweep, the commands that must notice, and its exceptions."""

    script: str
    #: Run in order; the first non-zero exit ends the site. Put the fastest and
    #: most sensitive first — for a checker with a self-test, that is the
    #: self-test.
    commands: tuple[tuple[str, ...], ...]
    #: Functions whose bodies are not swept, because they are the harness
    #: rather than the check: neutering a branch of a self-test measures the
    #: self-test against itself and reports whatever it likes.
    not_swept: tuple[str, ...]
    allowed: tuple[Allowed, ...]


TARGETS: tuple[Target, ...] = (
    Target(
        script="scripts/check_quality_claims.py",
        commands=(
            ("python3", "scripts/check_quality_claims.py", "--self-test"),
            ("python3", "scripts/check_quality_claims.py"),
        ),
        not_swept=("self_test", "stage_tree", "staged_run", "staged_process"),
        allowed=(
            Allowed(
                function="table_problems",
                source="expected is None",
                why="unreachable while the self-test's own first block holds. That block "
                "requires FIGURES to register a figure for every (series, corpus) pair in "
                "TABLE_ROWS and DIRECTIONAL_SERIES, so `figure_for` cannot return None here. "
                "It is a guard against a state a stronger assertion already refuses, and "
                "deleting it would turn that state from a message into a traceback",
            ),
            Allowed(
                function="main",
                source="args.self_test",
                why="no self-test can prove it was itself invoked. Neutered, `--self-test` "
                "runs the live check, which passes on a correct tree: both CI commands "
                "become the same command and nothing inside the file can notice. Closing it "
                "needs something outside the file to assert on what the run printed, which "
                "is a check on a string rather than on behaviour",
            ),
        ),
    ),
    # This file, measured by its own self-test. `main` is not swept: neutering
    # its dispatch makes `--self-test` run the live sweep, which sweeps this
    # target, which spawns `--self-test` again. That is the same bootstrap hole
    # the other target records in `allowed`, and here it also recurses, so it is
    # excluded rather than permitted.
    Target(
        script="scripts/check_assertions_can_fail.py",
        commands=(("python3", "scripts/check_assertions_can_fail.py", "--self-test"),),
        not_swept=("self_test", "main"),
        allowed=(),
    ),
)


@dataclass(frozen=True)
class Site:
    """One decision in a source file, and what neutering it looks like."""

    function: str
    lineno: int
    start: int
    end: int
    source: str
    replacement: str

    def describe(self) -> str:
        return f"{self.function}:{self.lineno} `{self.source}` -> {self.replacement}"


def sites(source: str, not_swept: tuple[str, ...]) -> list[Site]:
    """Every branch, loop and `problems +=` outside `not_swept`, innermost function wins."""
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(lineno: int, col: int) -> int:
        return offsets[lineno - 1] + col

    found: list[Site] = []

    def visit(node: ast.AST, function: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = function
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = child.name
            elif function and function not in not_swept:
                target = None
                if isinstance(child, ast.If):
                    target, replacement = child.test, "False"
                elif isinstance(child, ast.For):
                    target, replacement = child.iter, "[]"
                elif (
                    isinstance(child, ast.AugAssign)
                    and isinstance(child.target, ast.Name)
                    and child.target.id == "problems"
                ):
                    target, replacement = child.value, "[]"
                if target is not None:
                    start = offset(target.lineno, target.col_offset)
                    end = offset(target.end_lineno, target.end_col_offset)
                    found.append(
                        Site(
                            function=function,
                            lineno=child.lineno,
                            start=start,
                            end=end,
                            source=source[start:end],
                            replacement=replacement,
                        )
                    )
            visit(child, inner)

    visit(ast.parse(source), "")
    return found


def reddened(root: pathlib.Path, commands: tuple[tuple[str, ...], ...]) -> bool:
    """Whether any command comes back non-zero. A command that hangs counts as noticing."""
    for command in commands:
        argv = [sys.executable if part == "python3" else part for part in command]
        try:
            completed = subprocess.run(
                argv, cwd=root, capture_output=True, text=True, timeout=TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            return True
        if completed.returncode != 0:
            return True
    return False


def survivors(root: pathlib.Path, target: Target) -> tuple[list[Site], int]:
    """The sites that stay green when neutered, and how many were tried."""
    path = root / target.script
    source = path.read_text()
    swept = sites(source, target.not_swept)
    green: list[Site] = []
    try:
        for site in swept:
            path.write_text(source[: site.start] + site.replacement + source[site.end :])
            if not reddened(root, target.commands):
                green.append(site)
    finally:
        path.write_text(source)
    return green, len(swept)


def problems_for(root: pathlib.Path, target: Target) -> tuple[list[str], int, int]:
    """Everything wrong with one target: unallowed survivors, and allowances nothing needs."""
    green, tried = survivors(root, target)
    problems = []
    for site in green:
        if not any(a.function == site.function and a.source == site.source for a in target.allowed):
            problems.append(
                f"{target.script}: {site.describe()} leaves every command at exit 0. That line "
                f"can be deleted with the gate reporting itself green. Give the assertion a "
                f"case that only it can satisfy, or write the site into `allowed` with the "
                f"reason it cannot fail"
            )
    live = {(site.function, site.source) for site in green}
    every = {(site.function, site.source) for site in sites((root / target.script).read_text(), target.not_swept)}
    for allowance in target.allowed:
        key = (allowance.function, allowance.source)
        if key not in every:
            problems.append(
                f"{target.script}: `allowed` permits `{allowance.source}` in {allowance.function} "
                f"and there is no such branch any more. The code moved out from under the "
                f"permission: delete the entry or repoint it"
            )
        elif key not in live:
            problems.append(
                f"{target.script}: `allowed` permits `{allowance.source}` in {allowance.function} "
                f"and neutering it is now noticed. A permission that outlives what it permitted "
                f"widens this sweep without anyone deciding to: delete the entry"
            )
    return problems, len(green), tried


#: A script carrying one driven and one undriven site of each kind this sweep
#: enumerates. Both halves are load-bearing. The driven three prove the sweep
#: runs the commands and reads their exit codes; the undriven three prove it
#: enumerates `if`, `for` and `problems +=` at all. An earlier victim had only
#: `if` statements in it, and this sweep — run against itself — reported that
#: `isinstance(child, ast.For)` and the `problems +=` test could both be deleted
#: with its own self-test green, which would have narrowed what it looks for
#: without narrowing anything it says.
VICTIM = '''#!/usr/bin/env python3
import sys

DRIVEN_WORDS = ("driven-loop",)
UNDRIVEN_WORDS = ("undriven-loop",)


def driven_call(text):
    return [word for word in ("driven-call",) if word in text]


def undriven_call(text):
    return [word for word in ("undriven-call",) if word in text]


def problems_in(text):
    problems = []
    if "driven-if" in text:
        problems.append("driven-if")
    if "undriven-if" in text:
        problems.append("undriven-if")
    for word in DRIVEN_WORDS:
        problems.extend([word] if word in text else [])
    for word in UNDRIVEN_WORDS:
        problems.extend([word] if word in text else [])
    problems += driven_call(text)
    problems += undriven_call(text)
    return problems


def self_test():
    found = problems_in("driven-if driven-loop driven-call")
    if found != ["driven-if", "driven-loop", "driven-call"]:
        return 1
    return 0


def main():
    if "--self-test" in sys.argv:
        return self_test()
    return 0


sys.exit(main())
'''

#: In source order, which is the order `sites` returns them.
VICTIM_SITES = (
    '"driven-if" in text',
    '"undriven-if" in text',
    "DRIVEN_WORDS",
    "UNDRIVEN_WORDS",
    "driven_call(text)",
    "undriven_call(text)",
)
VICTIM_UNDRIVEN = ('"undriven-if" in text', "UNDRIVEN_WORDS", "undriven_call(text)")

VICTIM_TARGET = Target(
    script="victim.py",
    commands=(("python3", "victim.py", "--self-test"), ("python3", "victim.py")),
    not_swept=("self_test", "main"),
    allowed=(),
)


def with_allowed(allowed: tuple[Allowed, ...]) -> Target:
    """`VICTIM_TARGET` with a different exception list."""
    return Target(
        script=VICTIM_TARGET.script,
        commands=VICTIM_TARGET.commands,
        not_swept=VICTIM_TARGET.not_swept,
        allowed=allowed,
    )


def self_test() -> int:
    """Sweep a script whose uncovered assertions are known, and require exactly those."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "victim.py").write_text(VICTIM)

        found = tuple(site.source for site in sites(VICTIM, VICTIM_TARGET.not_swept))
        if found != VICTIM_SITES:
            print(
                f"self-test FAILED: the enumerated sites were {list(found)} and the victim "
                f"carries {list(VICTIM_SITES)}. A sweep that enumerates the wrong sites reports "
                f"on the wrong lines, and `not_swept` is what keeps the victim's own self-test "
                f"out of them",
                file=sys.stderr,
            )
            return 1

        problems, green, tried = problems_for(root, VICTIM_TARGET)
        if (tried, green) != (len(VICTIM_SITES), len(VICTIM_UNDRIVEN)):
            print(
                f"self-test FAILED: {tried} sites tried and {green} survived, and the victim "
                f"has {len(VICTIM_SITES)} sites of which {len(VICTIM_UNDRIVEN)} are undriven. "
                f"The driven half is what says the commands are run and their exit codes read",
                file=sys.stderr,
            )
            return 1
        for source in VICTIM_UNDRIVEN:
            if not any(source in problem for problem in problems):
                print(
                    f"self-test FAILED: the undriven site `{source}` was not reported: "
                    f"{problems}",
                    file=sys.stderr,
                )
                return 1
        if len(problems) != len(VICTIM_UNDRIVEN):
            print(
                f"self-test FAILED: {len(problems)} problems for {len(VICTIM_UNDRIVEN)} undriven "
                f"sites, so a driven one was reported as well: {problems}",
                file=sys.stderr,
            )
            return 1
        if (root / "victim.py").read_text() != VICTIM:
            print("self-test FAILED: the swept file was not put back", file=sys.stderr)
            return 1

        # The exception mechanism, and both ways an exception goes stale.
        permitted = with_allowed(
            tuple(Allowed("problems_in", source, "on purpose") for source in VICTIM_UNDRIVEN)
        )
        if problems_for(root, permitted)[0] != []:
            print(
                f"self-test FAILED: permitted survivors were still reported: "
                f"{problems_for(root, permitted)[0]}",
                file=sys.stderr,
            )
            return 1

        needless = with_allowed((Allowed("problems_in", '"driven-if" in text', "not needed"),))
        if not any("is now noticed" in problem for problem in problems_for(root, needless)[0]):
            print(
                f"self-test FAILED: an allowance for a branch that IS noticed was not reported: "
                f"{problems_for(root, needless)[0]}",
                file=sys.stderr,
            )
            return 1

        moved = with_allowed((Allowed("problems_in", '"deleted" in text', "gone"),))
        if not any("no such branch" in problem for problem in problems_for(root, moved)[0]):
            print(
                f"self-test FAILED: an allowance naming a branch that does not exist was not "
                f"reported: {problems_for(root, moved)[0]}",
                file=sys.stderr,
            )
            return 1

    print(
        f"self-test ok: over a staged script carrying a driven and an undriven `if`, `for` and "
        f"`problems +=`, the sweep enumerates all {len(VICTIM_SITES)} in source order, reports "
        f"the {len(VICTIM_UNDRIVEN)} undriven ones and no other, puts the file back, reports "
        f"nothing once they are permitted, and reports a permission both when the branch it "
        f"names is noticed after all and when the branch is gone; {len(TARGETS)} scripts are "
        f"swept for real"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    problems: list[str] = []
    counts = []
    for target in TARGETS:
        found, green, tried = problems_for(REPO_ROOT, target)
        problems += found
        counts.append((target.script, tried, green))

    if problems:
        print("FAIL: a check in this tree can be neutered and stay green:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    for script, tried, green in counts:
        print(
            f"ok: {script} — {tried} branches neutered one at a time, {tried - green} noticed by "
            f"the script's own commands and {green} recorded in `allowed` with a reason"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
