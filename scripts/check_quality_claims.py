#!/usr/bin/env python3
"""Fail when the two published pages disagree about how good these vectors are.

`README.md` and `description.yml` both tell a stranger what this embedder can
and cannot do, and `description.yml` is the copy that becomes the extension's
page in the DuckDB community registry — the one nobody arrives at through this
repository. Two prose copies of a quality claim are two chances to be wrong, and
this repository has already shipped that defect once on the function table,
which is why `check_documented_surface.py` exists.

A quality claim cannot be settled by loading the extension the way a function
signature can. What it can be settled against is the measurement that produced
it, so this holds three things at once:

WHAT IS ASSERTED
    1. Both files carry a section headed `SECTION_HEADING`, and neither is
       empty. A scan that reports clean because a heading was renamed out from
       under it is not a scan.
    2. Every quantity written in either section is one of `FIGURES` — a number
       somebody measured, recorded here with what it measures and where it came
       from. An unregistered quantity is a claim with nothing behind it, so the
       set is closed: a new figure reddens until it is entered here with its
       source.
    3. The two sections carry the same set of quantities. A figure in one and
       not the other is the defect this check is named for.
    4. Every string in `CLAIMS` appears in both sections. Assertion 3 compares
       quantities as a set, so on its own it cannot see two figures swapped
       between corpora; these pin each figure to the thing it is a figure of.
    5. No phrase in `BANNED` appears in either section — the hedges the figures
       replaced, and any vocabulary of speed. Speed was ruled out of the quality
       claim on purpose: the embedding-only number is an order of magnitude
       larger than the end-to-end one and would flatter the page.
    6. The revision of the bundled model is the revision the figures were
       measured on. Vectors from two model versions are not comparable, so a
       model bump does not make these figures stale quietly — it reddens here
       until the harness is re-run.
    7. Every entry in `FIGURES` is written on at least one of the two pages. A
       table of permitted values that outlives the sentence it permitted widens
       assertion 2 without anyone deciding to, which is the same shape as a
       mutation anchor naming a line that has been deleted.

WHAT IS NOT ASSERTED
    That the figures are true. `FIGURES` is a transcription of a measurement
    made elsewhere, named in each entry's `source`; nothing here re-runs it.
    Assertion 6 is what stops the transcription outliving the model it
    describes.

    A quality claim carrying no quantity at all. "Roughly as good as a
    transformer" in one file and not the other passes every assertion above.
    `BANNED` catches the specific hedges this page used to carry and no others.

Needs PyYAML, because `extended_description` is a block scalar inside a YAML
document and a regex cannot tell one from the prose around it.

    scripts/check_quality_claims.py
    scripts/check_quality_claims.py --self-test
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

README = "README.md"
DESCRIPTOR = "description.yml"
MODEL_SOURCE = "models/potion-base-8M/SOURCE.md"

#: The level-2 heading whose section carries the quality position, in both
#: files, spelled identically. Renaming it in one file is caught by assertion 1.
SECTION_HEADING = "What it is good at, and what it is not"

#: The bundled model revision the measurement in `FIGURES` was taken on, as
#: `models/potion-base-8M/SOURCE.md` records it. Assertion 6.
MEASURED_ON_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

FINETYPE_EVAL = "finetype eval/static-embedding-map-fidelity/results.json"
SWIFTEMBED = "SwiftEmbed, arxiv.org/abs/2510.24793"


@dataclass(frozen=True)
class Figure:
    """A quantity the pages are allowed to write, and what it is a figure of."""

    value: float
    what: str
    source: str


#: Every quantity either section may contain. Closed on purpose: a figure that
#: is not here reddens, so a new claim cannot reach the registry page without
#: someone writing down what it measures and who measured it.
FIGURES: list[Figure] = [
    # Ours. potion-base-8M against all-MiniLM-L6-v2, seed 42, 20 nearest
    # neighbours, UMAP(metric="cosine", n_neighbors=15, random_state=42).
    Figure(0.13185, "kNN overlap with MiniLM's map, long-form prose", FINETYPE_EVAL),
    Figure(0.27758, "kNN overlap with MiniLM's map, short text", FINETYPE_EVAL),
    Figure(0.40301, "kNN overlap with MiniLM's map, very short strings", FINETYPE_EVAL),
    Figure(0.71005, "cluster-structure retention vs MiniLM, long-form prose", FINETYPE_EVAL),
    Figure(0.66735, "cluster-structure retention vs MiniLM, short text", FINETYPE_EVAL),
    Figure(0.87503, "cluster-structure retention vs MiniLM, very short strings", FINETYPE_EVAL),
    Figure(0.39244, "potion-base-8M AMI over raw vectors, very short strings", FINETYPE_EVAL),
    Figure(0.35104, "all-MiniLM-L6-v2 AMI over raw vectors, very short strings", FINETYPE_EVAL),
    Figure(3000, "rows sampled from each of the two large corpora", FINETYPE_EVAL),
    Figure(216, "rows in the column-name corpus", FINETYPE_EVAL),
    Figure(12, "classes in the column-name corpus", FINETYPE_EVAL),
    Figure(20, "nearest neighbours compared; also the 20 Newsgroups corpus name", FINETYPE_EVAL),
    # Not ours. Published figures for the same model family, cited as such.
    Figure(0.901, "average precision on SprintDuplicateQuestions, potion-base-8M", SWIFTEMBED),
    Figure(0.847, "average precision on SprintDuplicateQuestions, Sentence-BERT", SWIFTEMBED),
    Figure(0.89, "low end of similarity and deduplication scores, as a share of SBERT", SWIFTEMBED),
    Figure(1.0, "high end of similarity and deduplication scores, as a share of SBERT", SWIFTEMBED),
    Figure(0.75, "classification, as a share of SBERT", SWIFTEMBED),
]

#: Assertion 4. Each pins a figure to what it is a figure of, so swapping two
#: between corpora reddens even though the set of quantities is unchanged.
#: Matched after whitespace is collapsed and case folded, so `description.yml`
#: wrapping one across two lines does not hide it.
CLAIMS: list[str] = [
    # AC4: the distinction the page was missing, in both files.
    "pairwise judgement",
    "ranked retrieval",
    # AC3: what it is good at, each with its figure.
    "90.1% average precision on SprintDuplicateQuestions where Sentence-BERT reports 84.7%",
    "89% to 100% of Sentence-BERT",
    "classification at about 75% of Sentence-BERT",
    "216 column names in 12 semantic classes",
    "0.3924 against 0.3510",
    # AC1: the figures that replaced "most" and "a minority".
    "71% of what MiniLM's map recovers on long-form prose, 67% on short text, "
    "88% on very short strings",
    "13% are the same rows on long-form prose, 28% on short text, 40% on very short strings",
    "20 nearest neighbours",
    # AC2: the direction of the shape dependence, not merely its existence.
    "worst on long prose and mildest on short strings",
]

#: Assertion 5. Case-folded substring match over the collapsed section.
BANNED: list[tuple[str, str]] = [
    ("a minority", "the hedge a measured figure replaced; it is compatible with 45%"),
    ("recovers most", "the hedge a measured figure replaced"),
    ("most of the cluster", "the hedge a measured figure replaced"),
    ("on the evidence we have", "a hedge standing where the corpus and the sample size belong"),
    ("faster", "speed is not part of the quality claim, deliberately"),
    ("speedup", "speed is not part of the quality claim, deliberately"),
    ("throughput", "speed is not part of the quality claim, deliberately"),
    ("latency", "speed is not part of the quality claim, deliberately"),
    ("per second", "speed is not part of the quality claim, deliberately"),
    ("rows/s", "speed is not part of the quality claim, deliberately"),
    ("×", "a multiplier is how a speed figure arrives; speed is ruled out here"),
]

#: A number in prose. The lookbehind keeps the `6` of `all-MiniLM-L6-v2` out,
#: and the lookahead keeps the `8` of `potion-base-8M` out while still admitting
#: `top-20` and `13%`. Thousands separators are kept and stripped when parsed.
QUANTITY = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%?)(?![\w-])")

FENCED = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
BARE_URL = re.compile(r"https?://\S+")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def strip_noise(text: str) -> str:
    """Remove everything in a section that is code or an address, not a claim.

    SQL examples are checked by `check_description_examples.py` against a real
    build, and a URL is not a quantity — `arxiv.org/abs/2510.24793` would
    otherwise read as one.
    """
    text = FENCED.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    text = LINK_TARGET.sub("]", text)
    text = BARE_URL.sub(" ", text)
    text = ISO_DATE.sub(" ", text)
    return text


def collapse(text: str) -> str:
    """Case-fold and squeeze whitespace, so a wrapped line matches an unwrapped one."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def section(text: str, heading: str) -> str | None:
    """The body under the level-2 `heading`, up to the next heading of that level or above.

    `None` when the heading is absent or written more than once — both are
    reported rather than treated as an empty section, because a section this
    cannot find is a section it is not checking.
    """
    lines = text.splitlines()
    starts = []
    for index, line in enumerate(lines):
        match = HEADING.match(line.strip())
        if match and len(match.group(1)) == 2 and match.group(2) == heading:
            starts.append(index)
    if len(starts) != 1:
        return None

    body = []
    for line in lines[starts[0] + 1 :]:
        match = HEADING.match(line.strip())
        if match and len(match.group(1)) <= 2:
            break
        body.append(line)
    return "\n".join(body)


def parse_quantity(digits: str, percent: str) -> tuple[float, int]:
    """A written quantity as (value, decimal places it was written to).

    `13%` is 0.13 written to two places; `0.3924` is itself written to four;
    `3,000` is three thousand written to none. The precision is returned so a
    figure can be compared at the precision the page rounded it to, which is how
    `13%` matches a measured 0.13185 and `14%` does not.

    A percentage carries two more places than its digits show: `13%` is 0.13,
    which is written to two decimal places, not to none.
    """
    places = len(digits.split(".")[1]) if "." in digits else 0
    value = float(digits.replace(",", ""))
    if percent:
        value /= 100.0
        places += 2
    return value, places


def quantities(section_text: str) -> list[tuple[float, int, str]]:
    """(value, decimal places, as written) for every quantity in a section."""
    found = []
    for match in QUANTITY.finditer(strip_noise(section_text)):
        value, places = parse_quantity(match.group(1), match.group(2))
        found.append((value, places, match.group(0).strip()))
    return found


def measured(value: float, places: int) -> Figure | None:
    """The registered figure a written quantity is a rounding of, if there is one.

    A written quantity matches when the measurement falls inside the interval
    that rounds to it at the precision it was written to — `13%` covers
    everything from 0.125 to 0.135, so it covers a measured 0.13185, and `14%`
    covers none of it. The interval is used rather than `round()` because
    `round()` has to pick a side of a tie and 0.13185 is one: at four places it
    is as good a rounding to 0.1318 as to 0.1319, and a check that admits one
    spelling and reddens on the other is arbitrary.
    """
    tolerance = 0.5 * 10.0**-places
    for figure in FIGURES:
        if abs(figure.value - value) <= tolerance + 1e-12:
            return figure
    return None


def unused_figures(*sections: str | None) -> list[str]:
    """Registered figures no section writes. Assertion 7."""
    written = [
        (value, places)
        for text in sections
        if text is not None
        for value, places, _ in quantities(text)
    ]
    problems = []
    for figure in FIGURES:
        if not any(
            abs(figure.value - value) <= 0.5 * 10.0**-places + 1e-12 for value, places in written
        ):
            problems.append(
                f"FIGURES registers {figure.value} ({figure.what}, from {figure.source}) and "
                f"neither page writes it. A permitted value nothing uses widens what the pages "
                f"may say without anyone deciding to: delete it, or put the claim back"
            )
    return problems


def region_problems(name: str, section_text: str | None) -> list[str]:
    """Everything wrong with one file's section, judged without the other file."""
    if section_text is None:
        return [
            f"{name} has no single `## {SECTION_HEADING}` section. It is one of the two files "
            f"that publish the quality position, and a section this cannot find is a section "
            f"it is not checking"
        ]
    collapsed = collapse(strip_noise(section_text))
    if not collapsed:
        return [
            f"{name}'s `## {SECTION_HEADING}` section is empty once code and links are removed; "
            f"an empty section agrees with anything"
        ]

    problems = []
    for value, places, written in quantities(section_text):
        if measured(value, places) is None:
            problems.append(
                f"{name} writes the quantity `{written}`, which is not one of the measured "
                f"figures in FIGURES. Either it is wrong, or it is a new claim and belongs in "
                f"FIGURES with what it measures and who measured it"
            )
    for claim in CLAIMS:
        if collapse(claim) not in collapsed:
            problems.append(f"{name}'s section does not carry the claim: {claim!r}")
    for phrase, reason in BANNED:
        if collapse(phrase) in collapsed:
            problems.append(f"{name}'s section contains {phrase!r} — {reason}")
    return problems


def disagreements(readme_text: str | None, descriptor_text: str | None) -> list[str]:
    """Quantities one section carries and the other does not."""
    if readme_text is None or descriptor_text is None:
        return []
    in_readme = {round(value, 6) for value, _, _ in quantities(readme_text)}
    in_descriptor = {round(value, 6) for value, _, _ in quantities(descriptor_text)}
    problems = []
    for value in sorted(in_readme - in_descriptor):
        problems.append(
            f"{README} claims {value} in its quality section and {DESCRIPTOR} does not. "
            f"{DESCRIPTOR} is the registry page; the two say the same thing about quality or "
            f"neither can be trusted"
        )
    for value in sorted(in_descriptor - in_readme):
        problems.append(
            f"{DESCRIPTOR} claims {value} in its quality section and {README} does not"
        )
    return problems


def revision_problems(source_text: str) -> list[str]:
    """The bundled model is the one the figures were measured on."""
    match = re.search(r"Revision:\s*`([0-9a-f]{40})`", source_text)
    if match is None:
        return [
            f"{MODEL_SOURCE} states no `Revision:` line. It is where the pinned model revision "
            f"is recorded, and without it nothing ties these figures to a model"
        ]
    if match.group(1) != MEASURED_ON_REVISION:
        return [
            f"the bundled model is at revision {match.group(1)} and the published figures were "
            f"measured on {MEASURED_ON_REVISION}. Vectors from two model versions are not "
            f"comparable: re-run the harness and update FIGURES, or the page describes a model "
            f"that is no longer in the binary"
        ]
    return []


def self_test() -> int:
    """Plant each defect this is meant to catch, and require it to be reported."""
    # The quantity parser: what it must see, and what it must not.
    seen = {written for _, _, written in quantities("13% and 0.3924 and 3,000 rows and 216 names")}
    if seen != {"13%", "0.3924", "3,000", "216"}:
        print(f"self-test FAILED: the quantity parser found {seen}", file=sys.stderr)
        return 1
    for identifier in ("potion-base-8M", "all-MiniLM-L6-v2", "`SELECT 41 + 1;`"):
        found = quantities(identifier)
        if found:
            print(
                f"self-test FAILED: {identifier} was read as a quantity: {found}", file=sys.stderr
            )
            return 1
    if quantities("[SwiftEmbed](https://arxiv.org/abs/2510.24793)"):
        print("self-test FAILED: a link target was read as a quantity", file=sys.stderr)
        return 1

    # Rounding: a page may round a measurement, and may not restate it.
    if measured(*parse_quantity("13", "%")) is None:
        print("self-test FAILED: 13% did not match a measured 0.13185", file=sys.stderr)
        return 1
    if measured(*parse_quantity("14", "%")) is not None:
        print("self-test FAILED: 14% matched a measured figure", file=sys.stderr)
        return 1
    if measured(*parse_quantity("45", "%")) is not None:
        print(
            "self-test FAILED: 45% matched a measured figure — that is the value 'a minority' "
            "was compatible with, and the reason this check exists",
            file=sys.stderr,
        )
        return 1
    if measured(*parse_quantity("0.132", "")) is None:
        print("self-test FAILED: 0.132 did not match a measured 0.13185", file=sys.stderr)
        return 1
    for tie in ("0.1318", "0.1319"):
        if measured(*parse_quantity(tie, "")) is None:
            print(
                f"self-test FAILED: {tie} did not match a measured 0.13185, which is exactly "
                "between the two at four places",
                file=sys.stderr,
            )
            return 1
    for outside in ("0.1317", "0.1320", "0.140"):
        if measured(*parse_quantity(outside, "")) is not None:
            print(f"self-test FAILED: {outside} matched a measured figure", file=sys.stderr)
            return 1

    # Section extraction, including the two ways it must refuse to report clean.
    document = (
        "# title\n\nintro\n\n"
        f"## {SECTION_HEADING}\n\nthe claim, at 13%.\n\n### a subsection\n\nstill in it, at 28%.\n\n"
        "## something else\n\nnot in it, at 99%.\n"
    )
    body = section(document, SECTION_HEADING)
    if body is None or "13%" not in body or "28%" not in body:
        print(f"self-test FAILED: the section body was {body!r}", file=sys.stderr)
        return 1
    if "99%" in body:
        print("self-test FAILED: the section ran past the next level-2 heading", file=sys.stderr)
        return 1
    if section(document.replace(f"## {SECTION_HEADING}", "## renamed"), SECTION_HEADING) is not None:
        print("self-test FAILED: a renamed heading was not reported", file=sys.stderr)
        return 1
    if section(document + f"\n## {SECTION_HEADING}\n\nagain\n", SECTION_HEADING) is not None:
        print("self-test FAILED: a duplicated heading was not reported", file=sys.stderr)
        return 1
    if region_problems("a_file", None) == []:
        print("self-test FAILED: a missing section reported clean", file=sys.stderr)
        return 1
    if region_problems("a_file", "\n") == []:
        print("self-test FAILED: an empty section reported clean", file=sys.stderr)
        return 1

    # A section built from the claims themselves, then broken one way at a time.
    good = " ".join(CLAIMS) + " 3,000 rows, 20 Newsgroups, 90.1%, 84.7%, 89% to 100%, 20."
    for label, text, needle in (
        ("a complete section reports nothing", good, None),
        (
            "an unmeasured quantity is caught",
            good + " and 45% of neighbours survive.",
            "not one of the measured figures",
        ),
        (
            "a dropped claim is caught",
            good.replace("pairwise judgement", "some other wording"),
            "does not carry the claim",
        ),
        (
            "two figures swapped between corpora are caught",
            good.replace(
                "13% are the same rows on long-form prose, 28% on short text",
                "28% are the same rows on long-form prose, 13% on short text",
            ),
            "does not carry the claim",
        ),
        ("a reinstated hedge is caught", good + " only a minority survive.", "a minority"),
        ("a speed figure is caught", good + " it is 397 times faster.", "faster"),
        ("a speed multiplier is caught", good + " a 397× speedup.", "speedup"),
    ):
        found = region_problems("a_file", text)
        if needle is None:
            if found:
                print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
                return 1
        elif not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    # The two files against each other. A figure in one and not the other is
    # the defect this check is named for, and it must be caught both ways round.
    if disagreements("13% and 28%", "13% and 28%") != []:
        print("self-test FAILED: two agreeing sections were reported", file=sys.stderr)
        return 1
    both_ways = disagreements("13% and 28%", "13%") + disagreements("13%", "13% and 28%")
    if len(both_ways) != 2:
        print(f"self-test FAILED: a one-sided figure gave {both_ways}", file=sys.stderr)
        return 1
    if disagreements("the overlap is 13%", "the overlap is 14%") == []:
        print("self-test FAILED: two sections disagreeing on a figure were not", file=sys.stderr)
        return 1

    # Assertion 7: a figure nothing writes any more.
    all_figures_written = " ".join(
        f"{figure.value}" for figure in FIGURES if figure.value not in (3000, 216, 12, 20)
    )
    all_figures_written += " 3,000 216 12 20"
    if unused_figures(all_figures_written) != []:
        print(
            f"self-test FAILED: a section writing every figure reported one unused: "
            f"{unused_figures(all_figures_written)}",
            file=sys.stderr,
        )
        return 1
    dropped = all_figures_written.replace("0.13185", "")
    if not any("0.13185" in problem for problem in unused_figures(dropped)):
        print(
            f"self-test FAILED: a figure no page writes was not reported: "
            f"{unused_figures(dropped)}",
            file=sys.stderr,
        )
        return 1

    # The model revision the figures belong to.
    if revision_problems(f"Revision: `{MEASURED_ON_REVISION}`") != []:
        print("self-test FAILED: the pinned revision was reported as a mismatch", file=sys.stderr)
        return 1
    if revision_problems("Revision: `" + "0" * 40 + "`") == []:
        print("self-test FAILED: a bumped model revision reported clean", file=sys.stderr)
        return 1
    if revision_problems("no revision line here") == []:
        print("self-test FAILED: a missing Revision: line reported clean", file=sys.stderr)
        return 1

    print(
        f"self-test ok: {len(FIGURES)} measured figures, {len(CLAIMS)} pinned claims and "
        f"{len(BANNED)} banned phrases; 13% rounds onto 0.13185 and 14% and 45% do not; a "
        f"renamed heading, an empty section, an unmeasured quantity, a dropped claim, two "
        f"figures swapped between corpora, a reinstated hedge, a speed figure, a one-sided "
        f"figure in either direction, a registered figure no page writes and a bumped model "
        f"revision are each reported"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    try:
        import yaml
    except ImportError:
        print(
            "PyYAML is not installed. `extended_description` is a block scalar inside a YAML "
            "document and a regex cannot tell one from the prose around it: pip install pyyaml",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []

    readme_path = REPO_ROOT / README
    descriptor_path = REPO_ROOT / DESCRIPTOR
    source_path = REPO_ROOT / MODEL_SOURCE
    for path in (readme_path, descriptor_path, source_path):
        if not path.is_file():
            print(f"{path} is not in the tree", file=sys.stderr)
            return 2

    descriptor = yaml.safe_load(descriptor_path.read_text())
    extended = (descriptor.get("docs") or {}).get("extended_description")
    if not extended:
        print(
            f"{DESCRIPTOR} has no docs.extended_description. It is the registry page's body and "
            f"the copy of the quality position a stranger reads",
            file=sys.stderr,
        )
        return 1

    readme_section = section(readme_path.read_text(), SECTION_HEADING)
    descriptor_section = section(extended, SECTION_HEADING)

    problems += region_problems(README, readme_section)
    problems += region_problems(f"{DESCRIPTOR} (extended_description)", descriptor_section)
    problems += disagreements(readme_section, descriptor_section)
    problems += unused_figures(readme_section, descriptor_section)
    problems += revision_problems(source_path.read_text())

    if problems:
        print("FAIL: the published quality position does not hold up:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    counted = len({round(value, 6) for value, _, _ in quantities(readme_section or "")})
    print(
        f"ok: {README} and {DESCRIPTOR} carry the same {counted} measured quantities under "
        f"`## {SECTION_HEADING}`, every one of them registered in FIGURES with its source, all "
        f"{len(CLAIMS)} pinned claims present in both, no banned phrase in either, every one "
        f"of the {len(FIGURES)} registered figures written on at least one page, and the "
        f"bundled model still at the revision they were measured on"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
