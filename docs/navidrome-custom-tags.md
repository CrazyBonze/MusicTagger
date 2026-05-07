# Navidrome Custom Tag Configuration

This document explains how to configure Navidrome to import the tags written by
musictagger. It covers which tags need explicit configuration, the aliases
required for each audio format, and a ready-to-paste `navidrome.toml` snippet.

## Background

Navidrome only imports a predefined set of tags by default. Tags outside that
set must be declared in `navidrome.toml` under the `Tags` section before
Navidrome will read or index them. After any change to tag configuration a
**full rescan** is required — a quick scan will not pick up the new definitions.

## Tags that need no configuration

The following musictagger tags map directly onto fields Navidrome already imports
by default. No extra configuration is needed for these.

| musictagger tag | Navidrome field | Formats covered by default |
|---|---|---|
| `bpm` | `bpm` | `TBPM` (ID3), `tmpo` (MP4), `bpm` (Vorbis/FLAC) |
| `mood` | `mood` | `TMOO` (ID3), `----:com.apple.iTunes:MOOD` (MP4), `MOOD` (Vorbis/FLAC) |
| `key` | `key` | `TKEY` (ID3), `----:com.apple.iTunes:initialkey` (MP4), `INITIALKEY` (Vorbis/FLAC) |

## Tags that require custom configuration

The remaining musictagger tags are 0–100 integer scores that are not part of
Navidrome's default tag set. They must be declared explicitly so that Navidrome
indexes them and makes them available as fields in Smart Playlists.

### How musictagger writes these tags

Each score tag is written under a format-specific field name:

| Format | Field name pattern | Example for `mood_happy` |
|---|---|---|
| ID3 (MP3, AIFF) | `TXXX:<NAME>` | `TXXX:MOOD_HAPPY` |
| Vorbis/FLAC/Ogg | `<NAME>` (upper-case) | `MOOD_HAPPY` |
| MP4/AAC | `----:com.apple.iTunes:<NAME>` | `----:com.apple.iTunes:MOOD_HAPPY` |
| APEv2 (WavPack) | `<NAME>` (upper-case) | `MOOD_HAPPY` |

Three tags use a different conventional name to match common tagger tooling:

| musictagger tag | Stored field name |
|---|---|
| `mood_dance` | `MOOD_DANCEABILITY` |
| `electronic` | `MOOD_ELECTRONIC` |
| `acoustic` | `MOOD_ACOUSTIC` |
| `instrumental` | `MOOD_INSTRUMENTAL` |

Navidrome aliases are case-insensitive, so a lowercase alias such as
`mood_happy` also matches the upper-case `MOOD_HAPPY` written by musictagger in
Vorbis and APEv2 files.

## navidrome.toml snippet

Paste the following into your `navidrome.toml`. The `Type = "float"` declaration
ensures Navidrome treats each tag as a number, enabling range comparisons in
Smart Playlists (e.g. `mood_happy > 70`).

```toml
# ── musictagger score tags ─────────────────────────────────────────────────────
#
# 0–100 integer scores written by musictagger. Declared here so Navidrome
# indexes them and exposes them as Smart Playlist fields.
#
# Aliases cover all three formats musictagger writes:
#   txxx:<name>                      → ID3 TXXX frame (MP3, AIFF)
#   <name>                           → Vorbis comment (FLAC, Ogg, WavPack)
#   ----:com.apple.itunes:<name>     → MP4 freeform atom (M4A, AAC)

Tags.mood_happy.Aliases = ["txxx:mood_happy", "mood_happy", "----:com.apple.itunes:mood_happy"]
Tags.mood_happy.Type = "float"

Tags.mood_sad.Aliases = ["txxx:mood_sad", "mood_sad", "----:com.apple.itunes:mood_sad"]
Tags.mood_sad.Type = "float"

Tags.mood_relaxed.Aliases = ["txxx:mood_relaxed", "mood_relaxed", "----:com.apple.itunes:mood_relaxed"]
Tags.mood_relaxed.Type = "float"

Tags.mood_aggressive.Aliases = ["txxx:mood_aggressive", "mood_aggressive", "----:com.apple.itunes:mood_aggressive"]
Tags.mood_aggressive.Type = "float"

Tags.mood_party.Aliases = ["txxx:mood_party", "mood_party", "----:com.apple.itunes:mood_party"]
Tags.mood_party.Type = "float"

# mood_dance is stored as MOOD_DANCEABILITY across all formats
Tags.mood_dance.Aliases = ["txxx:mood_danceability", "mood_danceability", "----:com.apple.itunes:mood_danceability"]
Tags.mood_dance.Type = "float"

# electronic, acoustic, and instrumental are stored as MOOD_<NAME> by musictagger.
# The non-prefixed aliases cover any files tagged by other tools without the prefix.
Tags.electronic.Aliases = ["txxx:mood_electronic", "mood_electronic", "----:com.apple.itunes:mood_electronic", "txxx:electronic", "electronic", "----:com.apple.itunes:electronic"]
Tags.electronic.Type = "float"

Tags.acoustic.Aliases = ["txxx:mood_acoustic", "mood_acoustic", "----:com.apple.itunes:mood_acoustic", "txxx:acoustic", "acoustic", "----:com.apple.itunes:acoustic"]
Tags.acoustic.Type = "float"

Tags.instrumental.Aliases = ["txxx:mood_instrumental", "mood_instrumental", "----:com.apple.itunes:mood_instrumental", "txxx:instrumental", "instrumental", "----:com.apple.itunes:instrumental"]
Tags.instrumental.Type = "float"

# timbre_brightness includes legacy "timbre" aliases for files tagged by older
# versions of musictagger that used the shorter field name.
Tags.timbre_brightness.Aliases = ["txxx:timbre_brightness", "timbre_brightness", "----:com.apple.itunes:timbre_brightness", "txxx:timbre", "timbre", "----:com.apple.itunes:timbre"]
Tags.timbre_brightness.Type = "float"

Tags.tonality.Aliases = ["txxx:tonality", "tonality", "----:com.apple.itunes:tonality"]
Tags.tonality.Type = "float"
```

## Docker

If you run Navidrome in Docker, place `navidrome.toml` in the host folder mapped
to `/data` inside the container. Navidrome automatically reads
`/data/navidrome.toml` on startup.

## After applying the configuration

1. Restart Navidrome so it loads the updated `navidrome.toml`.
2. Trigger a **full rescan** from the Navidrome UI (Settings → Scan Library →
   Full Scan). A quick scan will not process the new tag definitions.
3. The tags will appear as filterable and sortable fields under their configured
   names in Smart Playlists.
