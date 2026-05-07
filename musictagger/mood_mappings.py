"""Mapping from Genre Discogs400 class keys to human-readable mood/genre labels.

The Genre Discogs400 model uses class labels of the form ``"Genre---Subgenre"``
(e.g. ``"Electronic---Techno"``).  The default label derivation strips the
genre prefix and title-cases the subgenre.  For 16 subgenres that appear under
two different parent genres the bare subgenre name is ambiguous, so this dict
provides explicit override names for all 32 affected class keys.

For all other class keys (not present in this dict) callers should fall back to
stripping the ``"Genre---"`` prefix and title-casing the remainder.

Adding a new override is straightforward: insert a ``"Genre---Subgenre": "Name"``
entry here.  The key must match the class label exactly as it appears in the
model metadata JSON.
"""

from __future__ import annotations

# Maps Genre Discogs400 class key → display label.
# Only entries that require a non-default name are listed here.
GENRE_LABEL_OVERRIDES: dict[str, str] = {
    # Afrobeat — funk/soul vs jazz lineage
    "Funk / Soul---Afrobeat": "Afrobeat",
    "Jazz---Afrobeat": "Jazz Afrobeat",
    # Disco — electronic (Italo/Euro) vs funk/soul
    "Electronic---Disco": "Eurodisco",
    "Funk / Soul---Disco": "Disco",
    # Dub — electronic (dub techno) vs reggae
    "Electronic---Dub": "Dub Techno",
    "Reggae---Dub": "Dub",
    # Electro — pure electronic vs hip hop
    "Electronic---Electro": "Electro",
    "Hip Hop---Electro": "Electro Hip Hop",
    # Experimental — electronic vs rock
    "Electronic---Experimental": "Experimental Electronic",
    "Rock---Experimental": "Experimental Rock",
    # Gospel — folk/world/country vs funk/soul tradition
    "Folk, World, & Country---Gospel": "Gospel",
    "Funk / Soul---Gospel": "Soul Gospel",
    # Grime — electronic vs hip hop
    "Electronic---Grime": "Grime",
    "Hip Hop---Grime": "Hip Hop Grime",
    # Hardcore — electronic (gabber/speedcore) vs rock
    "Electronic---Hardcore": "Hardcore Techno",
    "Rock---Hardcore": "Hardcore",
    # Industrial — electronic vs rock
    "Electronic---Industrial": "Industrial",
    "Rock---Industrial": "Industrial Rock",
    # Neofolk — electronic vs rock
    "Electronic---Neofolk": "Neofolk",
    "Rock---Neofolk": "Neofolk Rock",
    # New Wave — synth-driven vs guitar-driven
    "Electronic---New Wave": "Synth New Wave",
    "Rock---New Wave": "New Wave",
    # Noise — electronic vs rock
    "Electronic---Noise": "Noise Electronic",
    "Rock---Noise": "Noise Rock",
    # Parody — pop vs rock
    "Pop---Parody": "Pop Parody",
    "Rock---Parody": "Parody",
    # Rhythm & Blues — blues-rooted vs funk/soul
    "Blues---Rhythm & Blues": "Blues R&B",
    "Funk / Soul---Rhythm & Blues": "R&B",
    # Ska — reggae vs rock (ska punk)
    "Reggae---Ska": "Ska",
    "Rock---Ska": "Ska Punk",
    # Trip Hop — electronic vs hip hop
    "Electronic---Trip Hop": "Trip Hop",
    "Hip Hop---Trip Hop": "Hip Hop Trip Hop",
}


def label_for_class(class_key: str) -> str:
    """Return the display label for a Genre Discogs400 class key.

    If *class_key* has an explicit entry in ``GENRE_LABEL_OVERRIDES`` that
    value is returned.  Otherwise the ``"Genre---"`` prefix is stripped and
    the remainder is title-cased.
    """
    if class_key in GENRE_LABEL_OVERRIDES:
        return GENRE_LABEL_OVERRIDES[class_key]
    subgenre = class_key.split("---", 1)[-1] if "---" in class_key else class_key
    return subgenre.title()
