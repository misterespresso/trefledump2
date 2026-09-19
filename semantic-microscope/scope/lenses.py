"""The lenses: 40 yes/no questions asked of every package summary.

Each one becomes a dimension of the embedding, and unlike dimension 247 of a
sentence transformer, each one is a sentence a person can read. They are grouped
only for the reader's benefit; the model sees them as independent questions asked
of the same state, which is what the API is built for.

Some of these are deliberately hard and subjective (humour, metaphor, "boring in a
reassuring way"). Those are the ones an interpretable axis earns its keep on, and
they are also the ones to distrust first.
"""

from __future__ import annotations

LENSES: dict[str, tuple[str, str]] = {
    # --- what kind of thing is it -------------------------------------------
    "is_library": ("form", "This is a library you import into your own code, rather than a program you run."),
    "is_cli": ("form", "This is a command-line tool you run from a terminal."),
    "is_framework": ("form", "This is a framework that wants to own the structure of your program."),
    "is_plugin": ("form", "This is a plugin, extension or backend for some other named tool."),
    "is_sdk": ("form", "This is a client or SDK for one specific external service or vendor."),
    "is_data": ("form", "What it ships is mainly data, assets or definitions rather than executable logic."),

    # --- what does it do ------------------------------------------------------
    "does_parsing": ("function", "Its job involves reading, parsing or writing a data format."),
    "does_network": ("function", "Its job involves talking over a network."),
    "does_math": ("function", "Its job involves numerical, scientific or statistical computation."),
    "does_testing": ("function", "Its job is testing, linting, type-checking or otherwise checking quality."),
    "does_packaging": ("function", "Its job is building, packaging, installing or deploying software."),
    "does_security": ("function", "Its job involves security, cryptography, authentication or certificates."),
    "does_text": ("function", "Its job involves manipulating text or strings."),
    "does_time": ("function", "Its job involves dates, times, scheduling or timezones."),
    "does_display": ("function", "Its job involves rendering, formatting or displaying output to a human."),

    # --- how does it talk about itself ---------------------------------------
    "claims_fast": ("rhetoric", "It claims to be fast, efficient or high-performance."),
    "claims_simple": ("rhetoric", "It claims to be simple, easy or lightweight."),
    "claims_best": ("rhetoric", "It claims to be the best, the only, or the definitive option."),
    "positions_against": ("rhetoric", "It defines itself against an existing alternative, as a replacement, "
                                      "drop-in, successor or improvement on something else."),
    "uses_superlative": ("rhetoric", "It uses a superlative or an intensifier such as 'blazing', 'ultimate', "
                                     "'fully' or 'the most'."),
    "names_a_vendor": ("rhetoric", "It names a specific company, product or brand."),
    "promises_safety": ("rhetoric", "It makes a promise about safety, correctness, reliability or standards "
                                    "compliance."),
    "is_playful": ("rhetoric", "There is humour, wordplay or a joke in how it is written."),
    "is_modest": ("rhetoric", "It undersells itself, describing itself as small, simple or unambitious."),
    "assumes_jargon": ("rhetoric", "Understanding it requires domain jargon that an outsider would not know."),
    "explains_purpose": ("rhetoric", "It says what the thing is *for*, rather than only what it is."),

    # --- how is the sentence built -------------------------------------------
    "is_full_sentence": ("style", "It is written as a complete sentence rather than a noun phrase."),
    "is_terse": ("style", "It is very short, under about ten words."),
    "has_acronym": ("style", "It contains an acronym or initialism."),
    "has_metaphor": ("style", "It uses a metaphor or a figurative image."),
    "is_imperative": ("style", "It is phrased as an instruction or imperative to the reader."),
    "mentions_python": ("style", "It mentions Python by name."),
    "lists_features": ("style", "It is largely a list of features separated by commas."),

    # --- what stance does it take --------------------------------------------
    "for_developers": ("stance", "Its audience is programmers rather than end users."),
    "is_infrastructure": ("stance", "It sounds like plumbing that most people will never interact with "
                                    "directly."),
    "is_low_level": ("stance", "It sounds like a small low-level building block rather than a complete "
                               "solution."),
    "is_batteries_included": ("stance", "It sounds like a complete, batteries-included solution."),
    "sounds_early": ("stance", "It sounds experimental, early or unfinished."),
    "sounds_boring": ("stance", "It sounds reassuringly boring: unglamorous, stable, unexciting work."),
    "hard_to_explain": ("stance", "Explaining what it does to a non-programmer would be hard."),
}

GROUPS = ("form", "function", "rhetoric", "style", "stance")


def questions() -> dict[str, dict]:
    """The Noul battery, in the shape the API takes."""
    return {key: {"type": "noul", "instructions": text} for key, (_, text) in LENSES.items()}


def state_for(pkg: dict) -> dict:
    """Only the pitch is shown. The name, topics and maturity are held back so they
    can be used as ground truth the model never saw."""
    return {
        "context": ("A one-line summary written by the authors of a Python package to describe it "
                    "on its listing page. Judge only this sentence."),
        "summary": pkg["summary"],
    }


if __name__ == "__main__":
    import collections
    print(f"{len(LENSES)} lenses")
    for g, n in collections.Counter(g for g, _ in LENSES.values()).items():
        print(f"  {g:10s} {n}")
    assert len(set(LENSES)) == len(LENSES)
    assert all(t.endswith(".") for _, t in LENSES.values())
