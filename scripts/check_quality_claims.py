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
it, and against the other copy of itself.

WHAT IS ASSERTED
    1. Both files carry a section headed `SECTION_HEADING`, and neither is
       empty. A scan that reports clean because a heading was renamed out from
       under it is not a scan.
    2. Every quantity written in either section is one of `FIGURES` — a number
       somebody measured, recorded here with what it measures and where it came
       from. An unregistered quantity is a claim with nothing behind it, so the
       set is closed: a new figure reddens until it is entered here with its
       source.
    3. The two sections write the same quantities, the same number of times, in
       the same order. Order matters because a figure moved to another row or
       another corpus leaves the set of quantities untouched: a summary-table
       cell swapped in one file and not the other used to pass a set comparison
       in both directions.
    4. Every string in `CLAIMS` appears in both sections. These pin the wording
       a figure is embedded in, so a rewrite that drops the corpus a figure
       belongs to reddens.
    5. The summary table's columns run `CORPORA` in that order, and every cell
       of a `TABLE_ROWS` row is the figure `FIGURES` registers for that row's
       series and that column's corpus. This is the assertion that survives a
       maintainer editing page and pin together: `13%` under `short text`
       reddens against a registered 0.27758 however the prose is worded.
    6. No sentence writes part of a series in `DIRECTIONAL_SERIES` without
       writing all of it. Both series run over the same three corpora and one of
       them is not monotone — region structure is 71%, 67%, 88%, so short text
       is the worst of the three and not the middle. A sentence quoting the two
       endpoints reads as a gradient and is not one, and this is what stops the
       page claiming a direction the reader cannot check from the same sentence.
    7. No universal quantifier appears in either section unless it is recorded
       in `ALLOWED_UNIVERSALS` with why it holds. See the block below for what
       this covers and what it cannot see; it is the widest of these assertions
       and the one with the largest blind spot.
    8. No phrase in `BANNED_IN_SECTION` appears in either section — the hedges
       the figures replaced. And no phrase in `BANNED_ON_PAGE` appears anywhere
       on either page: speed was ruled out of the product claim on purpose,
       because the embedding-only number is an order of magnitude larger than
       the end-to-end one, and a speed figure planted under *The SQL surface*
       flatters the product to the same reader as one planted here. That ban is
       page-wide for that reason, not section-wide, and it is the one scan that
       reads code as well as prose: `description.yml` publishes its SQL
       examples in fenced blocks, so a ban that ran after code was stripped was
       page-minus-code-wide and left the surface it was widened to cover open.
    9. The bundled model has not moved off `MEASURED_ON_REVISION`.
   10. Every entry in `FIGURES` and every entry in `ALLOWED_UNIVERSALS` is
       written on at least one of the two pages. A permitted value or a
       permitted universal that outlives the sentence it permitted widens
       assertions 2 and 7 without anyone deciding to, which is the same shape as
       a mutation anchor naming a line that has been deleted.

WHAT THE UNIVERSAL-QUANTIFIER SCAN COVERS, AND WHAT IT CANNOT SEE
    It covers exactly the words in `UNIVERSAL_WORDS`, matched whole, inside the
    two quality sections, after code spans and links are removed. It exists
    because both published defects it was written for were the same shape: a
    blanket sentence asserting a uniform property over a list whose items do not
    share it. "Every figure below compares potion-base-8M against MiniLM" stood
    over a section that deliberately mixes our measurement with a third party's,
    and cancelled the per-item sourcing the rest of the section does carefully.
    Pinning sentences in `CLAIMS` could not have found it, because `CLAIMS` pins
    what somebody thought to enumerate.

    It cannot see a universal written without one of those words. A bare plural
    generic — "the figures below compare X against Y" — asserts precisely the
    same thing as "every figure below compares X against Y" and carries no
    quantifier at all. That is the largest hole and it is not closeable by
    widening the word list; this scan is a floor, not a proof.

    It cannot see `invariably`, `in each case`, `without exception`, `bar none`,
    or any other spelling not in `UNIVERSAL_WORDS`. Adding one is a one-line
    change and should be made when one is met, rather than assumed absent.

    It does not judge truth. An entry in `ALLOWED_UNIVERSALS` is a person's
    recorded reason, not a derivation, and nothing here re-checks it. What it
    does guarantee is that the reason exists, in this file, next to the phrase
    it permits, where a reviewer reads it — and that it disappears when the
    sentence does, by assertion 10.

    It is scoped to the two quality sections and not to the whole page. The rest
    of `README.md` is full of true universals about a deterministic function
    ("`embed(NULL)` is always NULL"), and a scan that cried wolf there would be
    turned off. The speed ban in assertion 8 is the one that runs page-wide and
    into the code examples, because that is where its defects were planted.

WHAT IS NOT ASSERTED
    That the figures are true. `FIGURES` is a transcription of a measurement
    made elsewhere, named in each entry's `source`; nothing here re-runs it.

    That the figures were measured on the bundled revision. The harness in
    finetype downloads `potion-base-8M` by name and records no revision, and
    `results.json` carries none, so `MEASURED_ON_REVISION` is a tripwire and not
    a proof: it is the revision `SOURCE.md` recorded when these figures were
    published. Assertion 9 forces a model bump to be noticed and the measurement
    re-run or deliberately re-blessed. It cannot certify what the harness ran
    against, and the page does not claim it can.

    A quality claim carrying no quantity at all. "Roughly as good as a
    transformer" in one file and not the other passes every assertion above.
    `BANNED_IN_SECTION` catches the specific hedges this page used to carry and
    no others.

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
from collections import Counter
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

README = "README.md"
DESCRIPTOR = "description.yml"
MODEL_SOURCE = "models/potion-base-8M/SOURCE.md"

#: The level-2 heading whose section carries the quality position, in both
#: files, spelled identically. Renaming it in one file is caught by assertion 1.
SECTION_HEADING = "What it is good at, and what it is not"

#: The revision `models/potion-base-8M/SOURCE.md` recorded when the figures in
#: `FIGURES` were published. A tripwire, not a proof — see WHAT IS NOT ASSERTED.
MEASURED_ON_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"

FINETYPE_EVAL = "finetype eval/static-embedding-map-fidelity/results.json"
SWIFTEMBED = "SwiftEmbed, arxiv.org/abs/2510.24793"

#: The three corpora, in the order the summary table's columns run and in the
#: order the prose lists them. Assertion 5 reads the table against this, so a
#: reversed header row is a mismatch rather than an invisible edit.
CORPORA = ("long-form prose", "short text", "very short strings")


@dataclass(frozen=True)
class Figure:
    """A quantity the pages are allowed to write, and what it is a figure of."""

    value: float
    what: str
    source: str
    #: The named run of figures over `CORPORA` this belongs to, if any. Used by
    #: assertion 5 to read a summary-table cell and by assertion 6 to require a
    #: whole series in one sentence.
    series: str = ""
    #: Which of `CORPORA` this figure is for. Empty when the figure is not one
    #: of a per-corpus series.
    corpus: str = ""


#: Every quantity either section may contain. Closed on purpose: a figure that
#: is not here reddens, so a new claim cannot reach the registry page without
#: someone writing down what it measures and who measured it.
FIGURES: list[Figure] = [
    # Ours. potion-base-8M against all-MiniLM-L6-v2, seed 42, 20 nearest
    # neighbours, UMAP(metric="cosine", n_neighbors=15, random_state=42).
    Figure(
        0.13185,
        "kNN overlap with MiniLM's map, long-form prose",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="long-form prose",
    ),
    Figure(
        0.27758,
        "kNN overlap with MiniLM's map, short text",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="short text",
    ),
    Figure(
        0.40301,
        "kNN overlap with MiniLM's map, very short strings",
        FINETYPE_EVAL,
        series="kNN overlap",
        corpus="very short strings",
    ),
    Figure(
        0.71005,
        "cluster-structure retention vs MiniLM, long-form prose",
        FINETYPE_EVAL,
        series="region retention",
        corpus="long-form prose",
    ),
    Figure(
        0.66735,
        "cluster-structure retention vs MiniLM, short text",
        FINETYPE_EVAL,
        series="region retention",
        corpus="short text",
    ),
    Figure(
        0.87503,
        "cluster-structure retention vs MiniLM, very short strings",
        FINETYPE_EVAL,
        series="region retention",
        corpus="very short strings",
    ),
    Figure(0.39244, "potion-base-8M AMI over raw vectors, very short strings", FINETYPE_EVAL),
    Figure(0.35104, "all-MiniLM-L6-v2 AMI over raw vectors, very short strings", FINETYPE_EVAL),
    # Two entries for one value on purpose: the long-form and short-text corpora
    # are separate samples that happen to be the same size, and assertion 5
    # reads one table cell per corpus.
    Figure(
        3000,
        "rows sampled from the 20 Newsgroups posts",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="long-form prose",
    ),
    Figure(
        3000,
        "rows sampled from their subject lines",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="short text",
    ),
    Figure(
        216,
        "rows in the column-name corpus",
        FINETYPE_EVAL,
        series="corpus size",
        corpus="very short strings",
    ),
    Figure(12, "classes in the column-name corpus", FINETYPE_EVAL),
    Figure(20, "nearest neighbours compared; also the 20 Newsgroups corpus name", FINETYPE_EVAL),
    # Not ours. Published figures for the same model family, cited as such, and
    # measured against Sentence-BERT rather than against all-MiniLM-L6-v2.
    Figure(0.901, "average precision on SprintDuplicateQuestions, potion-base-8M", SWIFTEMBED),
    Figure(0.847, "average precision on SprintDuplicateQuestions, Sentence-BERT", SWIFTEMBED),
    Figure(0.89, "low end of similarity and deduplication scores, as a share of SBERT", SWIFTEMBED),
    Figure(1.0, "high end of similarity and deduplication scores, as a share of SBERT", SWIFTEMBED),
    Figure(0.75, "classification, as a share of SBERT", SWIFTEMBED),
]

#: Assertion 5. Summary-table row label → the series its cells are figures of.
#: The `corpus` row is prose rather than numbers and is pinned in `CLAIMS`.
TABLE_ROWS: dict[str, str] = {
    "rows": "corpus size",
    "nearest neighbours that survive": "kNN overlap",
    "region structure kept": "region retention",
}

#: Assertion 6. Series a reader could mistake for a gradient. `region retention`
#: is 71%, 67%, 88% — not monotone — and a sentence quoting 71% and 88% alone
#: told a reader with one-line descriptions they were at the good end when the
#: cited metric puts them at the worst. `corpus size` is left out: nobody claims
#: a direction over sample sizes, and requiring all three would redden the
#: sentence that says 216 rows is a small sample.
DIRECTIONAL_SERIES: tuple[str, ...] = ("kNN overlap", "region retention")

#: Assertion 4. Each pins a figure to what it is a figure of, or pins a
#: distinction the page would otherwise lose in a rewrite. Matched after
#: whitespace is collapsed and case folded, so `description.yml` wrapping one
#: across two lines does not hide it.
CLAIMS: list[str] = [
    # AC4: the distinction the page was missing, in both files, and the one
    # weakness that sits on the pairwise side of it.
    "pairwise judgement",
    "ranked retrieval",
    "a false duplicate, which is a pairwise failure and not a ranked-retrieval one",
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
    # AC2: the direction of the shape dependence for neighbourhoods, and the
    # fact that region structure does not follow it.
    "worst on long prose and mildest on very short strings",
    "13%, then 28%, then 40% as the text gets shorter",
    # The region sentence, tied to its corpora rather than to its conclusion.
    # Its trailing clause survives the first two figures being transposed, and
    # a transposed pair passes assertions 2, 3, 5 and 6 untouched: both values
    # stay registered, both files agree, the table is not edited and the series
    # is still quoted whole. What it leaves is a sentence that contradicts the
    # table two lines below it and tells a reader with one-line descriptions
    # the wrong end to be at, which is the defect this page was rewritten for.
    "71% on long-form prose, 67% on short text, 88% on very short strings",
    "short text is the weakest of the three for regions, not the middle of them",
    # The summary table's one prose row. Its numeric rows are read against
    # FIGURES by assertion 5, which a literal pin cannot do.
    "| corpus | 20 Newsgroups posts | their subject lines | column names |",
]

#: Assertion 8, inside the section. Case-folded substring match.
BANNED_IN_SECTION: list[tuple[str, str]] = [
    ("a minority", "the hedge a measured figure replaced; it is compatible with 45%"),
    ("recovers most", "the hedge a measured figure replaced"),
    ("most of the cluster", "the hedge a measured figure replaced"),
    ("on the evidence we have", "a hedge standing where the corpus and the sample size belong"),
]

#: Assertion 8, anywhere on either page. Speed is ruled out of the product claim
#: and a speed figure three sections down reaches the same reader.
BANNED_ON_PAGE: list[tuple[str, str]] = [
    ("faster", "speed is not part of the published claim, deliberately"),
    ("speedup", "speed is not part of the published claim, deliberately"),
    ("throughput", "speed is not part of the published claim, deliberately"),
    ("latency", "speed is not part of the published claim, deliberately"),
    ("per second", "speed is not part of the published claim, deliberately"),
    ("rows/s", "speed is not part of the published claim, deliberately"),
    ("×", "a multiplier is how a speed figure arrives; speed is ruled out here"),
]

#: Assertion 7. Words that assert a property over a whole set. Matched whole, so
#: `all-MiniLM-L6-v2` and `overall` are not hits — though both are usually
#: inside a code span and removed before this runs anyway.
UNIVERSAL_WORDS: tuple[str, ...] = (
    "every",
    "everything",
    "all",
    "always",
    "any",
    "anything",
    "each",
    "never",
    "only",
    "none",
)


@dataclass(frozen=True)
class Universal:
    """A universal these sections may write, and why it holds over its set."""

    phrase: str
    why: str


#: Assertion 7's exceptions, and the whole exception mechanism: a universal not
#: written out here reddens. Kept short on purpose — every entry is a sentence
#: nobody will re-derive, so the reason has to carry it.
ALLOWED_UNIVERSALS: list[Universal] = [
    Universal(
        "any phrase and its shuffle land in the same place",
        "permutation invariance is a property of the arithmetic mean rather than a sample of "
        "it. `crates/staticembed-core` pools by averaging token vectors; "
        "`the_pool_is_a_mean_so_order_is_lost_and_repetition_is_not` asserts it in Rust and "
        "`test/sql/06_text_the_tokenizer_treats_as_one_value.sql` asserts it against a loaded "
        "extension, so this quantifies over inputs nobody tried",
    ),
]

#: A number in prose. The lookbehind keeps the `6` of `all-MiniLM-L6-v2` out,
#: and the lookahead keeps the `8` of `potion-base-8M` out while still admitting
#: `top-20` and `13%`. Thousands separators are kept and stripped when parsed.
QUANTITY = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%?)(?![\w-])")

#: A whole word from `UNIVERSAL_WORDS`, over already-collapsed (case-folded) text.
UNIVERSAL = re.compile(r"(?<![\w-])(" + "|".join(UNIVERSAL_WORDS) + r")(?![\w-])")

#: A sentence end in collapsed prose. Trailing `*` and `_` are consumed so a
#: bolded lead-in ends its sentence where a reader sees it end.
SENTENCE_END = re.compile(r"(?<=[.;:!?])[*_]*\s+")

FENCED = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
BARE_URL = re.compile(r"https?://\S+")
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def strip_addresses(text: str) -> str:
    """Remove links, bare URLs and dates. Code is left where it is.

    An address is not a quantity — `arxiv.org/abs/2510.24793` and `2026-08-24`
    both parse as one — and neither is it a claim about the product. Code is a
    different matter: it is where a speed figure hides. This is what assertion
    8's page-wide half scans, so that half reads the SQL examples too.
    """
    text = LINK_TARGET.sub("]", text)
    text = BARE_URL.sub(" ", text)
    text = ISO_DATE.sub(" ", text)
    return text


def strip_noise(text: str) -> str:
    """Remove everything in a section that is code or an address, not a claim.

    SQL examples are checked by `check_description_examples.py` against a real
    build, so a number inside one is not a published figure.
    """
    text = FENCED.sub(" ", text)
    text = INLINE_CODE.sub(" ", text)
    return strip_addresses(text)


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


def rounds_onto(figure: Figure, value: float, places: int) -> bool:
    """Whether a quantity written to `places` places is a rounding of `figure`.

    The interval is used rather than `round()` because `round()` has to pick a
    side of a tie and 0.13185 is one: at four places it is as good a rounding to
    0.1318 as to 0.1319, and a check that admits one spelling and reddens on the
    other is arbitrary.
    """
    return abs(figure.value - value) <= 0.5 * 10.0**-places + 1e-12


def measured(value: float, places: int) -> Figure | None:
    """The registered figure a written quantity is a rounding of, if there is one.

    A written quantity matches when the measurement falls inside the interval
    that rounds to it at the precision it was written to — `13%` covers
    everything from 0.125 to 0.135, so it covers a measured 0.13185, and `14%`
    covers none of it.
    """
    for figure in FIGURES:
        if rounds_onto(figure, value, places):
            return figure
    return None


def figure_for(series: str, corpus: str) -> Figure | None:
    """The one registered figure for a series and a corpus. Assertion 5."""
    for figure in FIGURES:
        if figure.series == series and figure.corpus == corpus:
            return figure
    return None


def unused_figures(*sections: str | None) -> list[str]:
    """Registered figures no section writes. Assertion 10."""
    written = [
        (value, places)
        for text in sections
        if text is not None
        for value, places, _ in quantities(text)
    ]
    problems = []
    for figure in FIGURES:
        if not any(rounds_onto(figure, value, places) for value, places in written):
            problems.append(
                f"FIGURES registers {figure.value} ({figure.what}, from {figure.source}) and "
                f"neither page writes it. A permitted value nothing uses widens what the pages "
                f"may say without anyone deciding to: delete it, or put the claim back"
            )
    return problems


def unused_allowances(*sections: str | None) -> list[str]:
    """Permitted universals no section writes. Assertion 10."""
    haystack = " ".join(collapse(strip_noise(text)) for text in sections if text is not None)
    problems = []
    for allowance in ALLOWED_UNIVERSALS:
        if collapse(allowance.phrase) not in haystack:
            problems.append(
                f"ALLOWED_UNIVERSALS permits {allowance.phrase!r} and neither page writes it. A "
                f"live exception to the universal-quantifier scan that no longer covers any "
                f"sentence widens assertion 7 silently: delete it"
            )
    return problems


def table(section_text: str) -> list[list[str]]:
    """Every markdown table row in a section, as stripped cells, separators dropped."""
    rows = []
    for line in section_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("|")]
        if all(set(cell) <= set("-: ") and cell for cell in cells):
            continue
        rows.append(cells)
    return rows


def prose(section_text: str) -> str:
    """The section with its table rows removed, for assertions read per sentence."""
    kept = [
        line
        for line in section_text.splitlines()
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    ]
    return "\n".join(kept)


def table_problems(name: str, section_text: str) -> list[str]:
    """The summary table's cells are the figures they claim to be. Assertion 5."""
    rows = table(section_text)
    headers = [row for row in rows if row and row[0] == "" and tuple(row[1:]) == CORPORA]
    if len(headers) != 1:
        actual = [row[1:] for row in rows if row and row[0] == ""]
        return [
            f"{name}'s `## {SECTION_HEADING}` section has no single summary table whose columns "
            f"run {list(CORPORA)} in that order — found {actual}. Every figure below the header "
            f"is read by its column, so a reordered or missing header is not a cosmetic edit"
        ]
    width = len(CORPORA)

    problems = []
    for label, series in TABLE_ROWS.items():
        matches = [row for row in rows if row and row[0] == label]
        if len(matches) != 1:
            problems.append(
                f"{name}'s summary table has {len(matches)} rows labelled {label!r} and needs "
                f"exactly one: it is the row carrying the {series!r} figures"
            )
            continue
        cells = matches[0][1:]
        if len(cells) != width:
            problems.append(
                f"{name}'s summary table row {label!r} has {len(cells)} cells under "
                f"{width} corpora"
            )
            continue
        for corpus, cell in zip(CORPORA, cells):
            expected = figure_for(series, corpus)
            if expected is None:
                problems.append(
                    f"FIGURES registers no {series!r} figure for {corpus!r}, so the summary "
                    f"table's {label!r} row cannot be checked against it"
                )
                continue
            written = quantities(cell)
            if len(written) != 1:
                problems.append(
                    f"{name}'s summary table cell for {label!r} under {corpus!r} is {cell!r}, "
                    f"which carries {len(written)} quantities and needs exactly one"
                )
                continue
            value, places, as_written = written[0]
            if not rounds_onto(expected, value, places):
                problems.append(
                    f"{name}'s summary table puts `{as_written}` under {corpus!r} in the "
                    f"{label!r} row, and the measured {series} for {corpus!r} is "
                    f"{expected.value} ({expected.what}). Two cells swapped between corpora "
                    f"leave the set of quantities on the page unchanged, which is why this is "
                    f"read cell by cell"
                )
    return problems


def series_problems(name: str, section_text: str) -> list[str]:
    """No sentence writes part of a directional series. Assertion 6.

    The summary table is excluded: its rows are read cell by cell against
    `FIGURES` by assertion 5, which is a stronger statement than this one.
    """
    problems = []
    for sentence in SENTENCE_END.split(collapse(strip_noise(prose(section_text)))):
        for series in DIRECTIONAL_SERIES:
            members = [figure for figure in FIGURES if figure.series == series]
            written = {
                figure.corpus
                for value, places, _ in quantities(sentence)
                for figure in members
                if rounds_onto(figure, value, places)
            }
            if not written or len(written) == len(members):
                continue
            missing = [figure for figure in members if figure.corpus not in written]
            problems.append(
                f"{name} writes {len(written)} of the {len(members)} {series} figures in one "
                f"sentence and leaves out "
                + ", ".join(f"{figure.value} ({figure.corpus})" for figure in missing)
                + f". The sentence is: {sentence.strip()[:160]!r}. A partial series reads as a "
                f"direction, and {series} over {list(CORPORA)} is "
                + ", ".join(str(figure.value) for figure in members)
                + " — quote all three or the reader cannot tell a gradient from a dip"
            )
    return problems


def universal_problems(name: str, section_text: str) -> list[str]:
    """No unpermitted universal quantifier. Assertion 7."""
    collapsed = collapse(strip_noise(section_text))
    permitted = []
    for allowance in ALLOWED_UNIVERSALS:
        needle = collapse(allowance.phrase)
        start = collapsed.find(needle)
        while start != -1:
            permitted.append((start, start + len(needle)))
            start = collapsed.find(needle, start + 1)

    problems = []
    for match in UNIVERSAL.finditer(collapsed):
        if any(low <= match.start() and match.end() <= high for low, high in permitted):
            continue
        context = collapsed[max(0, match.start() - 60) : match.end() + 60]
        problems.append(
            f"{name}'s section writes the universal {match.group(1)!r} in: …{context}…  A "
            f"universal asserts one property over a whole set, and this section deliberately "
            f"mixes our measurement with a third party's and figures that compare nothing, so a "
            f"blanket sentence cancels the per-item sourcing the rest of it does. Rewrite it as "
            f"a bounded claim, or add it to ALLOWED_UNIVERSALS with why it holds"
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
    for phrase, reason in BANNED_IN_SECTION:
        if collapse(phrase) in collapsed:
            problems.append(f"{name}'s section contains {phrase!r} — {reason}")
    problems += table_problems(name, section_text)
    problems += series_problems(name, section_text)
    problems += universal_problems(name, section_text)
    return problems


def page_problems(name: str, page_text: str) -> list[str]:
    """Assertion 8's page-wide half: speed vocabulary anywhere on the page.

    Read over `strip_addresses` and not `strip_noise`, so fenced blocks and
    inline code spans are scanned as well as prose. `description.yml` publishes
    its SQL examples in fences and `README.md` carries six more, which is the
    surface this ban was widened to cover: a speed figure in a comment inside
    one reaches the same reader as a speed figure in a sentence.
    """
    collapsed = collapse(strip_addresses(page_text))
    problems = []
    for phrase, reason in BANNED_ON_PAGE:
        if collapse(phrase) in collapsed:
            position = collapsed.find(collapse(phrase))
            context = collapsed[max(0, position - 70) : position + 70]
            problems.append(
                f"{name} contains {phrase!r} — {reason}. Found in: …{context}…  This ban is "
                f"page-wide rather than section-wide: a speed figure under another heading "
                f"reaches the same reader"
            )
    return problems


def disagreements(readme_text: str | None, descriptor_text: str | None) -> list[str]:
    """Quantities the two sections do not both write, the same number of times, in order."""
    if readme_text is None or descriptor_text is None:
        return []
    in_readme = [(round(value, 6), written) for value, _, written in quantities(readme_text)]
    in_descriptor = [
        (round(value, 6), written) for value, _, written in quantities(descriptor_text)
    ]

    problems = []
    readme_counts = Counter(value for value, _ in in_readme)
    descriptor_counts = Counter(value for value, _ in in_descriptor)
    for value in sorted((readme_counts - descriptor_counts).elements()):
        problems.append(
            f"{README} claims {value} in its quality section and {DESCRIPTOR} does not, or does "
            f"not claim it as often. {DESCRIPTOR} is the registry page; the two say the same "
            f"thing about quality or neither can be trusted"
        )
    for value in sorted((descriptor_counts - readme_counts).elements()):
        problems.append(
            f"{DESCRIPTOR} claims {value} in its quality section and {README} does not, or does "
            f"not claim it as often"
        )
    if problems:
        return problems

    for index, (left, right) in enumerate(zip(in_readme, in_descriptor)):
        if left[0] != right[0]:
            problems.append(
                f"the two sections carry the same quantities in a different order: at position "
                f"{index + 1}, {README} writes `{left[1]}` where {DESCRIPTOR} writes "
                f"`{right[1]}`. A figure moved to another row or another corpus leaves the set "
                f"of quantities unchanged, which is how a swapped summary-table cell used to "
                f"pass this check in both directions"
            )
            break
    return problems


def revision_problems(source_text: str) -> list[str]:
    """The bundled model has not moved off the revision these figures were published against."""
    match = re.search(r"Revision:\s*`([0-9a-f]{40})`", source_text)
    if match is None:
        return [
            f"{MODEL_SOURCE} states no `Revision:` line. It is where the pinned model revision "
            f"is recorded, and without it nothing ties these figures to a model"
        ]
    if match.group(1) != MEASURED_ON_REVISION:
        return [
            f"the bundled model is at revision {match.group(1)} and the published figures were "
            f"transcribed against {MEASURED_ON_REVISION}. Vectors from two model versions are "
            f"not comparable: re-run the harness and update FIGURES, or re-bless them here "
            f"deliberately, rather than letting the page describe a model that is no longer in "
            f"the binary"
        ]
    return []


def self_test() -> int:  # noqa: C901
    """Plant each defect this is meant to catch, and require it to be reported.

    Each case plants one defect and names a needle the assertion it is written
    for produces. That is not a detail of style. A case whose needle some
    neighbouring assertion also satisfies proves nothing about its own, and
    this file shipped one: `("a reinstated hedge", good + " only a minority
    survive.", "a minority")` planted a hedge in a sentence carrying `only`, so
    the universal scan's message quoted the planted text back and satisfied the
    needle. `BANNED_IN_SECTION`'s enforcement could be deleted with every case
    here green.

    For the assertions named in `scripts/mutation_check.py` that is measured
    rather than asserted: each of those mutations disables one assertion and
    requires this self-test to redden. The rest of the assertions below have a
    case and no mutation, so for them this paragraph is a discipline and not a
    guarantee.
    """
    # FIGURES itself: assertion 5 reads one figure per (series, corpus), so two
    # entries claiming the same cell would make the table check pick arbitrarily.
    cells = [(figure.series, figure.corpus) for figure in FIGURES if figure.series]
    if len(cells) != len(set(cells)):
        print(f"self-test FAILED: FIGURES registers a (series, corpus) twice: {cells}", file=sys.stderr)
        return 1
    for series in (*TABLE_ROWS.values(), *DIRECTIONAL_SERIES):
        for corpus in CORPORA:
            if figure_for(series, corpus) is None:
                print(
                    f"self-test FAILED: FIGURES registers no {series!r} figure for {corpus!r}",
                    file=sys.stderr,
                )
                return 1

    # The quantity parser: what it must see, and what it must not.
    seen = {written for _, _, written in quantities("13% and 0.3924 and 3,000 rows and 216 names")}
    if seen != {"13%", "0.3924", "3,000", "216"}:
        print(f"self-test FAILED: the quantity parser found {seen}", file=sys.stderr)
        return 1
    for identifier in (
        "potion-base-8M",
        "all-MiniLM-L6-v2",
        "`SELECT 41 + 1;`",
        "```sql\nSELECT 41 + 1;\n```",
        "measured on 2026-08-24",
    ):
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
    # Both of these name the message they are written for. `== []` was enough
    # for the missing-section case and was not enough for the empty-section
    # one: an empty section fails every pin in CLAIMS as well, so the guard
    # could be deleted and this case still passed.
    if not any("it is not checking" in problem for problem in region_problems("a_file", None)):
        print(
            f"self-test FAILED: a missing section was not reported as missing: "
            f"{region_problems('a_file', None)}",
            file=sys.stderr,
        )
        return 1
    if not any("agrees with anything" in problem for problem in region_problems("a_file", "\n")):
        print(
            f"self-test FAILED: an empty section was not reported as empty: "
            f"{region_problems('a_file', chr(10))}",
            file=sys.stderr,
        )
        return 1

    # A summary table, then broken one cell and one header at a time.
    good_table = (
        "| | long-form prose | short text | very short strings |\n"
        "|---|---|---|---|\n"
        "| corpus | 20 Newsgroups posts | their subject lines | column names |\n"
        "| rows | 3,000 | 3,000 | 216 |\n"
        "| nearest neighbours that survive | 13% | 28% | 40% |\n"
        "| region structure kept | 71% | 67% | 88% |\n"
    )
    if table_problems("a_file", good_table) != []:
        print(
            f"self-test FAILED: a correct summary table was reported: "
            f"{table_problems('a_file', good_table)}",
            file=sys.stderr,
        )
        return 1
    for label, broken, needle in (
        (
            "two cells swapped between corpora",
            good_table.replace(
                "| nearest neighbours that survive | 13% | 28% | 40% |",
                "| nearest neighbours that survive | 28% | 13% | 40% |",
            ),
            "swapped between corpora",
        ),
        (
            "two cells transposed in the other row",
            good_table.replace(
                "| region structure kept | 71% | 67% | 88% |",
                "| region structure kept | 67% | 71% | 88% |",
            ),
            "swapped between corpora",
        ),
        (
            "a reversed header row",
            good_table.replace(
                "| | long-form prose | short text | very short strings |",
                "| | very short strings | short text | long-form prose |",
            ),
            "in that order",
        ),
        (
            "a deleted row",
            good_table.replace("| region structure kept | 71% | 67% | 88% |\n", ""),
            "needs exactly one",
        ),
        (
            "an emptied cell",
            good_table.replace(
                "| rows | 3,000 | 3,000 | 216 |", "| rows | 3,000 | | 216 |"
            ),
            "needs exactly one",
        ),
        ("no table at all", "prose with no table in it", "no single summary table"),
    ):
        found = table_problems("a_file", broken)
        if not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    # A whole series in one sentence, and the endpoints-only sentence that is
    # the defect: region structure is 71%, 67%, 88% and is not monotone.
    whole = (
        "It is worst on long prose and mildest on very short strings — 13%, then 28%, then 40% "
        "as the text gets shorter. Region structure is 71% on long-form prose, 67% on short "
        "text, 88% on very short strings."
    )
    if series_problems("a_file", whole) != []:
        print(
            f"self-test FAILED: two whole series were reported: "
            f"{series_problems('a_file', whole)}",
            file=sys.stderr,
        )
        return 1
    endpoints = "It is worst on long prose and mildest on short strings — 13% against 40% on neighbours, 71% against 88% on regions."
    found = series_problems("a_file", endpoints)
    if not any("kNN overlap" in problem for problem in found):
        print(f"self-test FAILED: an endpoints-only kNN sentence gave {found}", file=sys.stderr)
        return 1
    if not any("region retention" in problem for problem in found):
        print(f"self-test FAILED: an endpoints-only region sentence gave {found}", file=sys.stderr)
        return 1
    if series_problems("a_file", "216 rows is a small sample, and 3,000 is not.") != []:
        print(
            "self-test FAILED: corpus sizes were treated as a directional series; a sentence "
            "may cite one sample size without citing all three",
            file=sys.stderr,
        )
        return 1
    # A row writing two of the three region figures. The whole table cannot
    # fail this case — it writes every figure of both series, so it is not a
    # partial series however it is read — and a case that cannot fail is not a
    # case. The row is what `prose` has to remove.
    if series_problems("a_file", "| region structure kept | 71% | 67% |\n") != []:
        print(
            "self-test FAILED: a table row writing two of the three region figures was read as "
            "a sentence. The table's cells are checked against FIGURES by assertion 5, which is "
            "stronger, and a row is not a claim about a direction",
            file=sys.stderr,
        )
        return 1

    # The universal-quantifier scan, and its exception mechanism.
    for label, text, word in (
        ("every", "Every figure below compares the bundled model against MiniLM.", "every"),
        ("all", "All of these figures are ours.", "all"),
        ("only", "Only one of them is weak here.", "only"),
        ("never", "This never loses cluster structure.", "never"),
        ("everything", "Everything below that reads as a weakness is ranked retrieval.", "everything"),
        ("always", "It always returns a unit vector.", "always"),
        ("each", "Each figure here was measured on the bundled model.", "each"),
        ("none", "None of these came from a third party.", "none"),
    ):
        found = universal_problems("a_file", text)
        if not any(repr(word) in problem for problem in found):
            print(f"self-test FAILED: the universal {label!r} was not reported: {found}", file=sys.stderr)
            return 1
    if universal_problems("a_file", "the figures below compare this model against MiniLM") != []:
        print(
            "self-test FAILED: a sentence with no quantifier word was reported — the scan is "
            "documented as blind to a bare plural generic and must not pretend otherwise",
            file=sys.stderr,
        )
        return 1
    if universal_problems("a_file", "a vector from `all-MiniLM-L6-v2`") != []:
        print("self-test FAILED: a model name in a code span was read as a universal", file=sys.stderr)
        return 1
    for allowance in ALLOWED_UNIVERSALS:
        if universal_problems("a_file", allowance.phrase) != []:
            print(
                f"self-test FAILED: the permitted universal {allowance.phrase!r} was reported: "
                f"{universal_problems('a_file', allowance.phrase)}",
                file=sys.stderr,
            )
            return 1
        if universal_problems("a_file", allowance.phrase.replace("shuffle", "reversal")) == []:
            print(
                f"self-test FAILED: {allowance.phrase!r} was permitted after its wording changed, "
                f"so the allowance is matching more than the sentence it was written for",
                file=sys.stderr,
            )
            return 1
    if unused_allowances(" ".join(a.phrase for a in ALLOWED_UNIVERSALS)) != []:
        print("self-test FAILED: a written allowance was reported unused", file=sys.stderr)
        return 1
    if unused_allowances("nothing here quantifies over anything") == []:
        print(
            "self-test FAILED: an allowance no page writes reported clean; a dead exception "
            "widens the scan without anyone deciding to",
            file=sys.stderr,
        )
        return 1

    # A section built from the claims themselves, then broken one way at a time.
    good = (
        " ".join(CLAIMS)
        + " 3,000 rows, 20 Newsgroups, 90.1%, 84.7%, 89% to 100%, 20.\n"
        + good_table
        + "\nany phrase and its shuffle land in the same place.\n"
    )
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
            "two figures swapped between corpora in the prose are caught",
            good.replace(
                "13% are the same rows on long-form prose, 28% on short text",
                "28% are the same rows on long-form prose, 13% on short text",
            ),
            "does not carry the claim",
        ),
        (
            "two figures swapped between corpora in the table are caught",
            good.replace(
                "| nearest neighbours that survive | 13% | 28% | 40% |",
                "| nearest neighbours that survive | 28% | 13% | 40% |",
            ),
            "swapped between corpora",
        ),
        # Two hedges, each planted so that nothing but the BANNED_IN_SECTION
        # scan has anything to say about it: no universal, no quantity, no pin
        # broken. The first of these read "only a minority survive", and the
        # `only` meant the universal scan's message carried the needle `a
        # minority` in its quoted context — so the hedge ban could be deleted
        # and this whole suite stayed green. A needle a neighbouring assertion
        # can satisfy proves nothing about the assertion it is written for.
        ("a reinstated hedge is caught by the hedge ban", good + " a minority survive.", "a minority"),
        (
            "the hedge the region figures replaced is caught by the hedge ban",
            good + " it recovers most of the cluster structure.",
            "recovers most",
        ),
        (
            "the region sentence's two corpora swapped are caught",
            good.replace(
                "71% on long-form prose, 67% on short text",
                "71% on short text, 67% on long-form prose",
            ),
            "71% on long-form prose, 67% on short text",
        ),
        (
            "a blanket universal is caught",
            good + " Every figure below compares the bundled model against MiniLM.",
            "'every'",
        ),
    ):
        found = region_problems("a_file", text)
        if needle is None:
            if found:
                print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
                return 1
        elif not any(needle in problem for problem in found):
            print(f"self-test FAILED: {label} — got {found}", file=sys.stderr)
            return 1

    # Assertion 8's page-wide half. Its defect was a speed figure planted three
    # sections below the quality claim, where a section-scoped ban never looked.
    if page_problems("a_file", "a page with no speed claim on it") != []:
        print("self-test FAILED: a clean page was reported", file=sys.stderr)
        return 1
    for label, text in (
        ("a speed multiple", "## The SQL surface\n\nit is 397 times faster than a transformer."),
        ("a row rate", "## The SQL surface\n\nit embeds 50,000 rows per second."),
        ("a multiplier sign", "## The SQL surface\n\na 397× speedup."),
        ("a latency claim", "## Why this exists\n\nlatency is bounded."),
        # And the same figures written where the SQL examples live. That is the
        # surface this ban was widened to cover — `description.yml` publishes
        # every example in a fenced block and `README.md` carries six more —
        # and a ban read after `strip_noise` had already looked at none of it.
        (
            "a speed claim in a comment inside a fenced example",
            "## The SQL surface\n\n```sql\n-- 50,000 rows per second\nSELECT embed(name) FROM t;\n```",
        ),
        (
            "a speed claim inside an inline code span",
            "## The SQL surface\n\nthe comment reads `-- 397\u00d7 faster` in that example.",
        ),
    ):
        if page_problems("a_file", text) == []:
            print(f"self-test FAILED: {label} outside the quality section reported clean", file=sys.stderr)
            return 1
    # Reading code is not reading addresses. A URL that spells a banned word is
    # an address, and `strip_addresses` is the only thing keeping it out now
    # that this scan no longer runs after code has been removed.
    for label, text in (
        # A relative target, not an http one: `BARE_URL` would strip an http
        # target on its own, so a case built from one leaves LINK_TARGET with
        # nothing only it can do. README.md links `models/potion-base-8M/
        # SOURCE.md` this way.
        ("a relative link target", "see [the note](notes/10x-faster-embeddings.md)"),
        ("a bare URL", "see https://example.invalid/faster-than-sbert for the write-up"),
        ("a fenced example with no claim in it", "```sql\nSELECT embed('a') FROM t;\n```"),
    ):
        if page_problems("a_file", text) != []:
            print(
                f"self-test FAILED: {label} was read as a speed claim: "
                f"{page_problems('a_file', text)}",
                file=sys.stderr,
            )
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
    if disagreements("13% then 28%", "28% then 13%") == []:
        print(
            "self-test FAILED: the same figures in a different order reported clean — a cell "
            "moved to another row keeps the set unchanged and is the defect this must see",
            file=sys.stderr,
        )
        return 1
    if disagreements("3,000 and 3,000", "3,000") == []:
        print(
            "self-test FAILED: a figure written twice in one section and once in the other "
            "reported clean; the summary table writes 3,000 in two columns",
            file=sys.stderr,
        )
        return 1

    # Assertion 10: a figure nothing writes any more.
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
        f"self-test ok: {len(FIGURES)} measured figures, {len(CLAIMS)} pinned claims, "
        f"{len(TABLE_ROWS)} summary-table rows read cell by cell, {len(DIRECTIONAL_SERIES)} "
        f"series that must be quoted whole, {len(UNIVERSAL_WORDS)} scanned quantifiers with "
        f"{len(ALLOWED_UNIVERSALS)} recorded exception(s), {len(BANNED_IN_SECTION)} banned "
        f"hedges and {len(BANNED_ON_PAGE)} page-wide speed phrases; 13% rounds onto 0.13185 and "
        f"14% and 45% do not; a renamed heading, an empty section reported as empty rather "
        f"than as its missing claims, an unmeasured quantity, a dropped claim, two figures "
        f"swapped between corpora in prose and in the table, the region sentence's two corpora "
        f"swapped, a reversed table header, a deleted table row, an emptied cell, a series "
        f"quoted at its endpoints only, an unpermitted universal, a permitted universal whose "
        f"wording moved, an exception no page writes, a hedge reinstated in a sentence with "
        f"nothing else wrong with it, a speed figure outside the quality section and a speed "
        f"figure inside a fenced example, a one-sided figure in either direction, the same "
        f"figures in a different order, a figure written a different number of times, a "
        f"registered figure no page writes and a bumped model revision are each reported"
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

    readme_text = readme_path.read_text()
    readme_section = section(readme_text, SECTION_HEADING)
    descriptor_section = section(extended, SECTION_HEADING)

    problems += region_problems(README, readme_section)
    problems += region_problems(f"{DESCRIPTOR} (extended_description)", descriptor_section)
    problems += disagreements(readme_section, descriptor_section)
    problems += unused_figures(readme_section, descriptor_section)
    problems += unused_allowances(readme_section, descriptor_section)
    problems += page_problems(README, readme_text)
    problems += page_problems(f"{DESCRIPTOR} (extended_description)", extended)
    problems += revision_problems(source_path.read_text())

    if problems:
        print("FAIL: the published quality position does not hold up:", file=sys.stderr)
        for problem in problems:
            print(f"       {problem}", file=sys.stderr)
        return 1

    counted = len({round(value, 6) for value, _, _ in quantities(readme_section or "")})
    print(
        f"ok: {README} and {DESCRIPTOR} carry the same {counted} measured quantities under "
        f"`## {SECTION_HEADING}`, in the same order, every one of them registered in FIGURES "
        f"with its source; all {len(CLAIMS)} pinned claims present in both; every summary-table "
        f"cell the figure FIGURES registers for its row and column; no partial series in any "
        f"sentence; no universal quantifier outside the {len(ALLOWED_UNIVERSALS)} recorded in "
        f"ALLOWED_UNIVERSALS; no banned hedge in either section and no speed vocabulary anywhere "
        f"on either page, in its code examples as well as its prose; every one of the "
        f"{len(FIGURES)} registered figures written on at least "
        f"one page; and the bundled model still at the revision they were published against"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
