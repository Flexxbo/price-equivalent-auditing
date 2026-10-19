"""Cue catalog: all text-cue conditions used in the paper.

Each condition is one price-sweep curve (one PE). A condition appends `cued_text` to the cued
product's description and (optionally) `control_text` to the control product.

  - dose-response (social proof, 4 intensities): does the price premium rise
    monotonically with cue strength?  -> validates the metric as an instrument.
  - neutral control: an equal-length, non-persuasive factual sentence.
    Guards against "the model just prefers the product with an extra sentence."
  - vacuous: sentiment without information.
  - framing (constant information, different form): BOTH products carry the SAME
    fact in two forms; the cued one gets form A, the control gets form B. A nonzero
    premium here means the agent responds to form rather than information.

`type` tags let the analysis group conditions. Lengths are kept broadly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Cue:
    id: str
    family: str
    type: str          # "informative" | "neutral" | "vacuous" | "framing"
    cued_text: str
    control_text: str = ""   # empty -> control stays neutral
    intensity: int | None = None   # rank within a dose-response family
    note: str = ""


CUES: dict[str, Cue] = {c.id: c for c in [
    # --- dose-response: social proof, increasing intensity ---
    Cue("sp_weak", "social_proof", "informative",
        "A popular choice.", intensity=1),
    Cue("sp_medium", "social_proof", "informative",
        "Chosen by over 1,000 customers.", intensity=2),
    Cue("sp_strong", "social_proof", "informative",
        "Our best-selling model, chosen by over 10,000 customers.", intensity=3),
    Cue("sp_vstrong", "social_proof", "informative",
        "The #1 best-selling model, chosen by over 100,000 customers.", intensity=4),

    # --- other persuasion families (single-sentence, price/rating untouched) ---
    Cue("scarcity", "scarcity", "informative", "Only a few left in stock."),
    Cue("authority", "authority", "informative", "Recommended by audio experts."),
    # The default authority cue ('audio experts') is off-domain for coffee; this
    # domain-matched variant is used there (only the domain word changes).
    Cue("authority_coffee", "authority", "informative", "Recommended by coffee experts."),

    # --- controls ---
    Cue("neutral", "control", "neutral",
        "Each unit has a printed serial number on the housing.",
        note="valence-free filler"),
    Cue("vacuous", "control", "vacuous",
        "A product that people love.",
        note="sentiment, no verifiable information"),
    Cue("price_only", "control", "none", "",
        note="no cue on either product; matched pair, only price differs "
             "(price-sensitivity calibration). run_m4.py sets no_cue for type 'none'."),

    # --- neutral-sentence probe: do genuinely value-free facts vary in premium? ---
    Cue("neutral_color", "control", "neutral", "Available in a black finish.",
        note="value-neutral attribute"),
    Cue("neutral_manual", "control", "neutral",
        "A printed user manual is included in the box.",
        note="value-neutral inclusion"),
    Cue("neutral_barcode", "control", "neutral",
        "The packaging carries a printed barcode.",
        note="maximally sterile identifier"),
    Cue("neutral_eco", "control", "neutral", "Ships in recyclable packaging.",
        note="valence probe: sustainability-coded sentence; analysed separately "
             "(see the stimulus neutrality appendix)"),

    # --- framing at constant information (control carries the same fact, other form) ---
    Cue("framing_pct_frac", "framing", "framing",
        cued_text="Chosen by 25% of shoppers in this category.",
        control_text="Chosen by 1 in 4 shoppers in this category.",
        note="25% == 1 in 4; identical information, different form"),
    Cue("framing_rating", "framing", "framing",
        cued_text="Rated 4.8 out of 5 by buyers.",
        control_text="Rated 9.6 out of 10 by buyers.",
        note="4.8/5 equals 9.6/10: identical statistic, different scale"),
    # stronger framing: gain vs loss framing of the SAME fact (90% keep / 10% return)
    Cue("framing_gainloss", "framing", "framing",
        cued_text="9 out of 10 buyers keep it.",
        control_text="1 out of 10 buyers returns it.",
        note="90% keep == 10% return; gain vs loss framing -- the strong framing test"),

    # --- length/presence controls: BOTH products carry a matched-length sentence, so
    #     a premium reflects the cue's CONTENT, not the mere presence/length of a
    #     sentence on one side (addresses the length confound in the neutral condition) ---
    Cue("sp_strong_lenctrl", "length_control", "control",
        cued_text="Our best-selling model, chosen by over 10,000 customers.",
        control_text="This model carries a printed identification code on its housing.",
        note="strong social proof vs a length-matched neutral sentence -> isolates content"),
    Cue("presence_ctrl", "length_control", "control",
        cued_text="This model carries a printed identification code on its housing.",
        control_text="This model comes in a plain protective retail carton.",
        note="two matched-length neutral sentences -> PE ~0 if mere presence/length is inert"),

    # === Cue-selection catalog (mo_*): 3 Cialdini families x 3 strengths ===
    # All stipulated true for the synthetic experiment. Strength varies on ONE axis per
    # family: social proof = crowd size + intensity phrase; authority = credibility of the
    # endorser (form constant); scarcity = remaining-quantity urgency. Numbers are true
    # product attributes and are the only numeric tokens the gate will allow the rewriter.
    Cue("mo_sp_weak", "social_proof", "informative",
        "A popular choice, chosen by over 10 customers.", intensity=1,
        note="cue selection: stipulated-true social proof, weak"),
    Cue("mo_sp_med", "social_proof", "informative",
        "Very popular, chosen by over 1,000 customers.", intensity=2,
        note="cue selection: stipulated-true social proof, medium"),
    Cue("mo_sp_strong", "social_proof", "informative",
        "Super popular, chosen by over 10,000 customers.", intensity=3,
        note="cue selection: stipulated-true social proof, strong"),
    Cue("mo_auth_weak", "authority", "informative",
        "Recommended by a store associate.", intensity=1,
        note="cue selection: authority, low-credibility endorser"),
    Cue("mo_auth_med", "authority", "informative",
        "Recommended by audio experts.", intensity=2,
        note="cue selection: authority, domain experts"),
    Cue("mo_auth_strong", "authority", "informative",
        "Recommended by leading audio engineers and sound professionals.", intensity=3,
        note="cue selection: authority, high-credibility endorser"),
    Cue("mo_scar_weak", "scarcity", "informative",
        "Only 10 left in stock.", intensity=1,
        note="cue selection: scarcity, mild"),
    Cue("mo_scar_med", "scarcity", "informative",
        "Only 5 left in stock.", intensity=2,
        note="cue selection: scarcity, medium"),
    Cue("mo_scar_strong", "scarcity", "informative",
        "Almost sold out — only 2 left in stock.", intensity=3,
        note="cue selection: scarcity, strong (urgency phrase + smallest count)"),
]}

# The nine cues of the cue-selection experiment (3 families x 3 strengths).
MINOPT_CUES = [
    "mo_sp_weak", "mo_sp_med", "mo_sp_strong",
    "mo_auth_weak", "mo_auth_med", "mo_auth_strong",
    "mo_scar_weak", "mo_scar_med", "mo_scar_strong",
]

# Main cue battery (run by run_m4.py; 'M4' is the internal label of the main audit).
M4_CONDITIONS = ["sp_weak", "sp_medium", "sp_strong", "sp_vstrong",
                 "scarcity", "authority",
                 "neutral", "vacuous",
                 "framing_pct_frac", "framing_rating", "framing_gainloss",
                 "sp_strong_lenctrl", "presence_ctrl"]
DOSE_FAMILY = ["sp_weak", "sp_medium", "sp_strong", "sp_vstrong"]


def get_cue(cue_id: str) -> Cue:
    if cue_id not in CUES:
        raise KeyError(f"unknown cue '{cue_id}'. Known: {list(CUES)}")
    return CUES[cue_id]


if __name__ == "__main__":
    for cid in M4_CONDITIONS:
        c = get_cue(cid)
        ctl = f"  | control: \"{c.control_text}\"" if c.control_text else ""
        print(f"{cid:18s} [{c.type:11s}] \"{c.cued_text}\"{ctl}")
