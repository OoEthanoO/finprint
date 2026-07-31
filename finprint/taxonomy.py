"""Broad taxonomic grouping for the 32 WMMS species.

Species-level accuracy is ~0.79, but 89% of the model's mistakes stay inside the
correct family, so the coarse group is right ~0.98 of the time (see
reports/error_analysis.md). Reporting the group alongside the species gives an
answer that is usually right even when the species guess is not.

Three groups, split the way the acoustics and the taxonomy agree:
  toothed whale (odontocetes: dolphins, pilot/killer whales, beluga, narwhal,
                 sperm whale) — clicks, whistles, burst pulses
  baleen whale  (mysticetes) — low-frequency moans and song
  pinniped      (seals, walrus) — highly variable, often low-frequency
"""

from __future__ import annotations

TOOTHED_WHALE = "toothed whale"
BALEEN_WHALE = "baleen whale"
PINNIPED = "pinniped"

# Keyed by the dataset's own label strings (models/labels.json).
GROUPS: dict[str, str] = {
    # pinnipeds
    "Bearded_Seal": PINNIPED,
    "Harp_Seal": PINNIPED,
    "Leopard_Seal": PINNIPED,
    "Ross_Seal": PINNIPED,
    "Walrus": PINNIPED,
    "Weddell_Seal": PINNIPED,
    # baleen whales
    "Bowhead_Whale": BALEEN_WHALE,
    "Fin,_Finback_Whale": BALEEN_WHALE,
    "Humpback_Whale": BALEEN_WHALE,
    "Minke_Whale": BALEEN_WHALE,
    "Northern_Right_Whale": BALEEN_WHALE,
    "Southern_Right_Whale": BALEEN_WHALE,
    # toothed whales
    "Atlantic_Spotted_Dolphin": TOOTHED_WHALE,
    "Beluga,_White_Whale": TOOTHED_WHALE,
    "Bottlenose_Dolphin": TOOTHED_WHALE,
    "Clymene_Dolphin": TOOTHED_WHALE,
    "Common_Dolphin": TOOTHED_WHALE,
    "False_Killer_Whale": TOOTHED_WHALE,
    "Frasers_Dolphin": TOOTHED_WHALE,
    "Grampus,_Rissos_Dolphin": TOOTHED_WHALE,
    "Killer_Whale": TOOTHED_WHALE,
    "Long-Finned_Pilot_Whale": TOOTHED_WHALE,
    "Melon_Headed_Whale": TOOTHED_WHALE,
    "Narwhal": TOOTHED_WHALE,
    "Pantropical_Spotted_Dolphin": TOOTHED_WHALE,
    "Rough-Toothed_Dolphin": TOOTHED_WHALE,
    "Short-Finned_Pacific_Pilot_Whale": TOOTHED_WHALE,
    "Sperm_Whale": TOOTHED_WHALE,
    "Spinner_Dolphin": TOOTHED_WHALE,
    "Striped_Dolphin": TOOTHED_WHALE,
    "White-beaked_Dolphin": TOOTHED_WHALE,
    "White-sided_Dolphin": TOOTHED_WHALE,
}


def group_of(species: str) -> str | None:
    """Broad group for a species label, or None if it isn't one we know."""
    return GROUPS.get(species)
