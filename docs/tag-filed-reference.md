# Music Tagger — Tag Field Reference

This document describes every metadata field written by the music tagger, how each one is computed, how reliable it is, and how to use it effectively when building smart playlists or browsing a large library in Navidrome or compatible software.

Each field includes a **reliability rating** on a five-star scale that reflects real-world accuracy across a wide range of genres — not just the easy cases.

---

## Quick Reference

Fields marked **built-in** are indexed by Navidrome automatically with no configuration. Fields marked **custom** must be declared in `navidrome.toml` as described in the Navidrome configuration guide — without this, they will not appear in Smart Playlists.

| Field | Navidrome Field Name | Config | Type | Range / Format | Reliability |
|---|---|---|---|---|---|
| `bpm` | `bpm` | built-in | Integer | 30 – 286 | ★★★★☆ |
| `mood` | `mood` | built-in | Text | Three sub-genre labels | ★★★☆☆ |
| `mood_happy` | `mood_happy` | custom | Score | 0 – 100 | ★★★★☆ |
| `mood_sad` | `mood_sad` | custom | Score | 0 – 100 | ★★★★☆ |
| `mood_relaxed` | `mood_relaxed` | custom | Score | 0 – 100 | ★★★★☆ |
| `mood_aggressive` | `mood_aggressive` | custom | Score | 0 – 100 | ★★★★★ |
| `mood_party` | `mood_party` | custom | Score | 0 – 100 | ★★★★☆ |
| `mood_dance` | `mood_dance` | custom | Score | 0 – 100 | ★★★★☆ |
| `acoustic` | `acoustic` | custom | Score | 0 – 100 | ★★★★☆ |
| `electronic` | `electronic` | custom | Score | 0 – 100 | ★★★☆☆ |
| `instrumental` | `instrumental` | custom | Score | 0 – 100 | ★★★☆☆ |
| `key` | `key` | built-in | Text | e.g. `C major`, `A minor` | ★★★★☆ |
| `timbre_brightness` | `timbre_brightness` | custom | Score | 0 – 100 | ★★☆☆☆ |
| `tonality` | `tonality` | custom | Score | 0 – 100 | ★★☆☆☆ |

---

## Understanding Score Fields

All fields listed with a `0 – 100` range are **probability scores**, not subjective ratings. A score of `72` for `mood_relaxed` means the model estimated a 72% probability that this track belongs to the "relaxed" class. These are not calibrated the same way across all fields — some span the full range confidently (like `mood_aggressive`), while others cluster in a narrow band and are less useful for filtering (like `timbre_brightness`).

A safe general approach: use **threshold filtering** rather than exact values. "Tracks where `mood_aggressive` > 60" is a meaningful query. "Tracks where `timbre_brightness` > 50" is not.

---

## Field Reference

---

### `bpm`
**Beats Per Minute**

**Navidrome field:** `bpm` — built-in, no configuration required.

**How it's computed:** Two-model ensemble. DeepRhythm (a PyTorch CNN) produces the primary estimate alongside a confidence score. When confidence falls below threshold, TempoCNN is invoked as a secondary backend — both models' results are then scored against TempoCNN's internal probability distribution and the higher-confidence candidate wins. Audio is sampled from the middle 180 seconds of the track to avoid intros and outros skewing the result.

**How to use it:**
- Filter by BPM range to build energy-matched playlists (e.g. running mixes at 150–175, dinner music at 60–100).
- Sort a queue from low to high BPM for a gradual warm-up or wind-down.
- Combine with `mood_dance` to distinguish genuinely danceable fast tracks from fast-but-not-groovy ones (metal at 150 BPM vs. dance music at 150 BPM will score very differently on `mood_dance`).

**Pitfalls:**
- Tracks with **variable tempo**, rubato passages, or no clear pulse (some ambient, classical, experimental) will produce unreliable readings. The number will be technically plausible but not meaningful.
- Metal and ambient genres specifically were identified as edge cases for DeepRhythm during development — the TempoCNN fallback exists precisely to correct the half/double-tempo errors these genres cause. If you notice BPM values that seem exactly half or double what you'd expect, these genres are the most likely source.
- The 180-second clip means very short tracks (under ~30 seconds) use the full available audio, which may be less reliable.
- `bpm` is an integer — sub-BPM precision is discarded.

**Reliability: ★★★★☆**
Strong for most recorded music with a clear beat. Less trustworthy on genres without a consistent pulse.

---

### `mood`
**Sub-genre Classification (Top 3)**

**Navidrome field:** `mood` — built-in, no configuration required.

**How it's computed:** Runs the Discogs-EffNet pipeline through the Genre Discogs400 classifier — a model trained on 3.3 million Discogs tracks across 400 music style labels (ROC-AUC 0.95). The three highest-scoring sub-genre labels are extracted and written as a semicolon-separated text value (e.g. `Psychedelic Rock; Prog Rock; Acid Rock`). The full list of all 400 possible labels, organised by parent genre, is in the [Appendix: Genre Discogs400 Label Taxonomy](#appendix-genre-discogs400-label-taxonomy) at the end of this document.

**How to use it:**
- The most powerful field for **music discovery**. Filter by sub-genre to surface similar music across your library.
- Use partial text matching — searching for `Ambient` will catch tracks tagged `Ambient; New Age; Celtic`, `Ambient; Experimental; Downtempo`, and so on.
- Particularly useful for finding what you're in the mood for when you know the style but not the artist.
- Note that tracks can appear in multiple genres, which is intentional — a Pink Floyd track might correctly be `Folk; Ambient; Pop Rock` at once.

**Pitfalls:**
- The model classifies based on **acoustic features**, not knowledge of the artist. It cannot know a band's genre by reputation — it infers from sound. This means acoustically atypical tracks get unexpected labels.
- **Cover albums and arrangements** (lullaby versions, orchestral covers, etc.) tend to be classified by their sonic character rather than the original genre. A lullaby version of a rock song will come back as Ambient or Experimental.
- The model occasionally pattern-matches on **texture over context**: a slow, melancholy Pink Floyd track may get tagged Doom Metal because it shares sonic qualities without being the genre. Treat fringe results as suggestions rather than facts.
- Labels are **sub-genre strings**, not controlled vocabulary — you may see both `Pop Rock` and `Pop-Rock` style variants depending on what the Discogs taxonomy uses. Check your library software's text matching behaviour.
- Songs that are genuinely genre-defying or highly experimental produce the least reliable results.

**Reliability: ★★★☆☆**
Excellent for well-recorded music in mainstream genres. Degrades gracefully on edge cases but can produce surprising results that require a tolerance for occasional misses.

---

### `mood_happy`
**Happiness / Positivity Score**

**Navidrome field:** `mood_happy` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `happy`. Score represents probability (0–100) that the track belongs to the happy class.

**How to use it:**
- High scores (70+) correlate with bright, major-key, energetic music. Good for morning playlists or "feel-good" queues.
- Combining `mood_happy > 60` with `mood_aggressive < 20` filters for genuinely pleasant rather than intense tracks.
- Inversely useful: `mood_happy < 20` is a reasonable proxy for "heavy or dark music."

**Pitfalls:**
- Can conflate **energy** with happiness. A very fast metal track may score slightly higher than expected because energy is a shared acoustic feature. Always combine with `mood_aggressive` to disambiguate.
- Does not distinguish between **types** of positivity — a triumphant anthem and a bubbly pop song both score high.

**Reliability: ★★★★☆**
Consistent and well-calibrated. Works best in combination with other mood scores.

---

### `mood_sad`
**Sadness / Melancholy Score**

**Navidrome field:** `mood_sad` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `sad`.

**How to use it:**
- High scores (70+) characterise slow, minor-key, or emotionally heavy music. Use for "late night" or introspective playlists.
- Often correlates with high `mood_relaxed` — melancholy tracks tend to be slow. Use both together when the distinction matters (a relaxed-but-not-sad playlist should filter `mood_sad < 40`).

**Pitfalls:**
- `mood_sad` and `mood_happy` are **not opposites** and do not sum to 100. A track can score 80 on both. This happens with emotionally complex music — bittersweet, nostalgic, or anthemic songs register high on both scales simultaneously, which is actually a musically reasonable outcome.
- Very aggressive music tends to score low on both, rather than high on sad.

**Reliability: ★★★★☆**
Well-calibrated. The non-exclusive relationship with `mood_happy` is a feature once you understand it, not a bug.

---

### `mood_relaxed`
**Relaxation / Calmness Score**

**Navidrome field:** `mood_relaxed` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `relaxed`.

**How to use it:**
- One of the most practically useful fields. High scores (80+) reliably identify calm, low-energy music — suitable for focus, sleep, background listening.
- `mood_relaxed > 70` combined with `acoustic > 50` is an effective filter for unplugged/acoustic calm music.
- `mood_relaxed > 80` with `instrumental > 60` finds ambient and instrumental calm music without vocals.

**Pitfalls:**
- Slow tempo is a strong acoustic signal for this classifier, so **slow but intense** music (slow doom metal, for example) may score higher than feels appropriate. Cross-reference with `mood_aggressive` to catch these.
- Very long, drifting tracks with ambient sections may score high even if the broader album is energetic.

**Reliability: ★★★★☆**
Among the more reliable mood scores. The signal is strong and the results are consistently useful for building low-energy playlists.

---

### `mood_aggressive`
**Aggression / Intensity Score**

**Navidrome field:** `mood_aggressive` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `aggressive`.

**How to use it:**
- The highest-confidence field in the entire set. Scores above 80 reliably identify hard rock, metal, punk, hardcore, and similarly intense music.
- Ideal for workout and high-energy playlists. `mood_aggressive > 75` is a clean filter for "intense music."
- Inversely, `mood_aggressive < 15` filters for everything that is not harsh or intense — useful as a negative constraint when you want to exclude heavy music from a playlist without specifying exactly what you want.

**Pitfalls:**
- **Distortion** is a strong signal for this classifier. Clean but very fast music (some electronic, drum and bass) scores lower than distorted-but-mid-tempo music. It is measuring harshness as much as intensity.
- Very short, sharp percussion-heavy tracks may score high even if the overall feel is not aggressive.

**Reliability: ★★★★★**
The best-performing field in the set. Tested across metal, pop, ambient, and everything in between — it holds up consistently. Use it with confidence.

---

### `mood_party`
**Party Energy Score**

**Navidrome field:** `mood_party` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `party`.

**How to use it:**
- High scores (80+) identify music associated with social, celebratory, or high-energy crowd contexts. ABBA, Cascada, and upbeat dance tracks score highest.
- Best used in combination with `mood_dance` — party energy without danceability suggests music that's exciting to hear but hard to move to; both fields high together confirms genuinely floor-filling music.

**Pitfalls:**
- The classifier was trained on stylistic associations with party contexts, not BPM or beat strength alone. Some genres score high for cultural reasons — a track may score high on `mood_party` because its genre is associated with parties even if the specific track is a slower ballad from that artist.
- Not a reliable substitute for `mood_dance` when actual danceability is what you want.

**Reliability: ★★★★☆**
Reliable for its intended purpose. Combine with `mood_dance` for best results.

---

### `mood_dance`
**Danceability Score**

**Navidrome field:** `mood_dance` — requires custom configuration in `navidrome.toml`. Note: the underlying tag is stored as `MOOD_DANCEABILITY` in audio files across all formats; the `navidrome.toml` aliases map this back to the `mood_dance` field name.

**How it's computed:** Binary Discogs-EffNet classifier. Note: stored internally as `mood_danceability` in some tag formats (ID3, MP4) for compatibility with Jaikoz and MusicBrainz Picard tooling.

**How to use it:**
- Measures how well-suited a track is for dancing — accounts for beat regularity, groove, and rhythmic character, not just tempo.
- Very useful as a differentiator: a 150 BPM metal track and a 150 BPM dance track will score very differently here.
- `mood_dance > 70` is a tight filter for music you can actually move to, rather than music that is merely fast.
- Slow tracks can still score moderately high if they have a strong groove (soul, slow R&B, etc.).

**Pitfalls:**
- Does not capture **dance style** — a waltz and a drum-and-bass track could theoretically both score high by different routes. Use `bpm` and `mood` sub-genres to further narrow genre-specific dance styles.
- Slower remixes or extended versions of danceable tracks may score lower than the original due to reduced beat density.

**Reliability: ★★★★☆**
Solid and practically useful. Works well in combination with BPM for playlist construction.

---

### `acoustic`
**Acoustic Character Score**

**Navidrome field:** `acoustic` — requires custom configuration in `navidrome.toml`. Stored in audio files as `MOOD_ACOUSTIC`; the aliases in `navidrome.toml` cover both the prefixed and unprefixed form.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `acoustic`. Measures the presence of acoustic (non-electronic, non-amplified) sound sources.

**How to use it:**
- High scores (70+) reliably identify music dominated by acoustic instruments — folk, classical, acoustic guitar, piano pieces, choral music.
- `acoustic > 65` combined with `instrumental > 60` is an effective filter for instrumental acoustic music (background, study, classical).
- Contrast with `electronic` — the two fields are complementary but not mutually exclusive (a track can have both acoustic and electronic elements).

**Pitfalls:**
- Heavily **produced** or **compressed** acoustic music may score lower than expected because the production signature masks the acoustic source quality.
- The model responds to tonal character, so lightly distorted electric guitar can read as semi-acoustic.

**Reliability: ★★★★☆**
Generally consistent and useful. Accurate for the obvious cases and reasonable on ambiguous ones.

---

### `electronic`
**Electronic Character Score**

**Navidrome field:** `electronic` — requires custom configuration in `navidrome.toml`. Stored in audio files as `MOOD_ELECTRONIC`; the aliases in `navidrome.toml` cover both the prefixed and unprefixed form.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `electronic`.

**How to use it:**
- Use to find synthesizer-heavy, produced, or electronic music. High scores (75+) reliably identify dance, ambient electronic, synth-pop, and similar genres.
- `electronic > 60` combined with `mood_dance > 50` targets electronic dance music specifically.

**Pitfalls:**
- This is the least consistent of the three character fields. The classifier appears to respond to **synthesizer-like timbres and production density** rather than strict genre membership. This means:
  - Densely produced music with lush arrangements (even if not electronic in origin) may score unexpectedly high.
  - Some 1970s artists using early synthesisers will score higher than you'd expect for their overall genre.
- `acoustic` and `electronic` can both be high on the same track — this is not a bug. It reflects music with mixed character, like an acoustic guitar recorded with heavy electronic production.

**Reliability: ★★★☆☆**
Reliable for clearly electronic music but shows false positives on produced or orchestral recordings. Cross-reference with `mood` sub-genres for confirmation.

---

### `instrumental`
**Instrumental Score (Absence of Vocals)**

**Navidrome field:** `instrumental` — requires custom configuration in `navidrome.toml`. Stored in audio files as `MOOD_INSTRUMENTAL`; the aliases in `navidrome.toml` cover both the prefixed and unprefixed form.

**How it's computed:** Binary Discogs-EffNet classifier. The positive class is `instrumental` (absence of significant vocal content), not `voice`. A high score means more instrumental; a low score means vocal-forward.

**How to use it:**
- `instrumental > 75` is a reliable filter for music without prominent lead vocals — useful for focus playlists, background music, or classical/jazz without distracting lyrics.
- Combine with `acoustic > 60` and `mood_relaxed > 60` for study/work playlists.
- A moderate score (40–60) often indicates music with occasional vocals, choral elements, or where the vocals are more textural than lyrical.

**Pitfalls:**
- **Heavily distorted, screamed, or growled vocals** (as found in extreme metal) are difficult for the classifier to distinguish from guitar noise. This produces inflated instrumental scores on vocal-heavy metal tracks — Slayer, for example, scores 44–72% instrumental despite prominent vocals throughout. If you are building a "no vocals" playlist and your library includes extreme metal, you may get false positives.
- Sparse vocal arrangements (a single spoken word, background choir) may tip the score lower than the music's overall character suggests.
- Does not distinguish between different vocal roles: lead vocals, harmonies, and choir all count the same.

**Reliability: ★★★☆☆**
Works well for most genres. The distorted-vocal problem is a known, specific failure mode that primarily affects extreme metal and some punk/hardcore. If your library doesn't include these genres heavily, the score is more trustworthy.

---

### `key`
**Musical Key**

**Navidrome field:** `key` — built-in, no configuration required.

**How it's computed:** Essentia's KeyExtractor algorithm — a classical HPCP (Harmonic Pitch Class Profile) analysis. No machine learning model required; this is a deterministic signal processing algorithm. Audio is decoded at 44100 Hz (full bandwidth) specifically for this computation.

**Format:** A string such as `C major`, `F# minor`, `Bb major`.

**How to use it:**
- Useful for **DJs and mix planning** — filter for harmonically compatible keys using the Circle of Fifths (adjacent and relative keys mix well).
- Can help identify tracks that will clash harmonically when played in sequence.
- Build playlists in a single key for a cohesive tonal feel.
- Some library tools support Camelot notation for harmonic mixing — the key string can be converted externally if needed.

**Pitfalls:**
- Tracks with **no clear tonal centre** (atonal, noise, or purely rhythmic music) will produce an arbitrary key result that should be disregarded. Check `tonality` — if that score is low, treat `key` as unreliable for that track.
- **Modulating tracks** (songs that change key mid-way) produce a result based on the most dominant key detected, which may not represent the full track.
- Tracks with heavy pitch effects, extreme distortion, or very low recording quality can produce incorrect results.
- The algorithm is designed for Western tonality and may be unreliable on non-Western music with alternative tuning systems.

**Reliability: ★★★★☆**
One of the more technically sound fields — it uses a well-established algorithm rather than a learned model. Reliable for conventionally tonal music. Disregard on atonal, noise, or rhythmically-only content.

---

### `timbre_brightness`
**Timbre Brightness Score**

**Navidrome field:** `timbre_brightness` — requires custom configuration in `navidrome.toml`. Legacy aliases for the shorter `TIMBRE` field name are included in the config to cover files tagged by older versions of the tool.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `bright`. Intended to measure whether the track's timbral character is bright (treble-forward, sharp) or dark (bass-forward, muted).

**How to use it:**
- In principle, high scores indicate treble-rich music (bright cymbals, sharp transients, high-frequency content); low scores indicate warm, bass-heavy, or muted recordings.
- Could theoretically be used to adjust for listening environment (e.g. avoiding overly bright music on harsh speakers).

**Pitfalls:**
- **In practice, this field currently shows very limited range.** Scores across vastly different genres (metal, ambient, pop, lullabies) cluster in the 43–54 range. This compressed distribution means the field provides almost no useful signal for filtering or sorting.
- The likely cause is that mastering levels and loudness normalisation mask the underlying timbral differences the model was trained to detect.
- Do not use this field as a primary filter until a broader distribution is confirmed on your library.

**Reliability: ★★☆☆☆**
The model exists and runs successfully, but the output distribution is too narrow to be practically useful for most collections. Monitor this field on future batches to see if the pattern holds.

---

### `tonality`
**Tonal vs. Atonal Score**

**Navidrome field:** `tonality` — requires custom configuration in `navidrome.toml`.

**How it's computed:** Binary Discogs-EffNet classifier, positive class = `tonal`. A high score means the model considers the track to have clear tonal organisation (melody, harmony, recognisable key); a low score suggests an atonal or arhythmic character.

**How to use it:**
- Intended as a quality gate on `key` — if `tonality` is very low, the detected `key` is unreliable.
- In principle, could be used to filter out highly experimental, noise, or atonal content.

**Pitfalls:**
- **This field currently shows anomalous results.** Scores for clearly tonal music (Beatles, ABBA, Enya) sit in the 0–73 range, with many tracks scoring in single digits. Music that is unambiguously tonal should score near 100 on this scale, so something about the model's output for this collection is miscalibrated.
- The root cause is unclear — it may be a model version mismatch, an issue with the positive class index, or a feature of how the EffNet embeddings interact with this specific classifier on well-produced commercial recordings.
- Do not use this field for filtering until you can verify it produces expected results on a known-tonal track in your library.

**Reliability: ★★☆☆☆**
Technically functional but producing suspicious output on this collection. Treat as unreliable until further investigation. The `key` field itself is more trustworthy than this score currently suggests.

---

## Using Fields Together: Suggested Combinations

The fields are most powerful in combination. A few starting points:

**Study / Focus Playlist**
`mood_relaxed > 65`, `mood_aggressive < 20`, `instrumental > 55`

**Workout / High Energy**
`mood_aggressive > 60` OR (`bpm > 140` AND `mood_dance > 50`), `mood_relaxed < 30`

**Background Dinner Music**
`acoustic > 50`, `mood_relaxed > 55`, `mood_aggressive < 25`, `bpm` between 70–120

**Electronic Dance**
`electronic > 65`, `mood_dance > 65`, `bpm` between 120–145

**Late Night / Melancholy**
`mood_sad > 55`, `mood_relaxed > 50`, `mood_aggressive < 20`

**Genre Exploration**
Use `mood` text search to surface all tracks tagged with a specific sub-genre across artists and albums.

---

## Notes on Model Provenance

All score fields use the **Discogs-EffNet** pipeline — a shared audio embedding model trained on Discogs data, with lightweight classifier heads per tag. The embeddings are computed once per track and all classifier heads run from the same representation. The genre (`mood`) field uses a separate Discogs Genre400 classifier head trained on 400 style labels from 3.3 million tracks.

BPM uses **DeepRhythm** as the primary model with **TempoCNN (deepsquare-k16)** as a scoring arbiter on low-confidence tracks. Key detection uses **Essentia's built-in KeyExtractor** with no learned model.

This means fields that share the EffNet backbone can have **correlated errors** — if the embedding misrepresents a track (unusual recording quality, non-standard production), multiple fields may be affected simultaneously. It also means all EffNet-based fields benefit equally from improvements to the shared extractor in future model versions.

---

## Appendix: Genre Discogs400 Label Taxonomy

The `mood` field draws from the 400 sub-genre labels below, organised by Discogs parent genre. When filtering with text search in Navidrome Smart Playlists, use the sub-genre label exactly as it appears here — these are the strings written to the `mood` tag (e.g. `mood contains "Doom Metal"`, `mood contains "Bossa Nova"`).

Note that sub-genre names are **not unique across parent genres** — for example, `Disco` appears under both Electronic and Funk/Soul, `Gospel` under both Folk/World/Country and Funk/Soul, and `Trip Hop` under both Electronic and Hip Hop. The classifier assigns the label based on the acoustic character of the track, regardless of which parent genre it came from.

### Blues
Boogie Woogie, Chicago Blues, Country Blues, Delta Blues, Electric Blues, Harmonica Blues, Jump Blues, Louisiana Blues, Modern Electric Blues, Piano Blues, Rhythm & Blues, Texas Blues

### Brass & Military
Brass Band, Marches, Military

### Children's
Educational, Nursery Rhymes, Story

### Classical
Baroque, Choral, Classical, Contemporary, Impressionist, Medieval, Modern, Neo-Classical, Neo-Romantic, Opera, Post-Modern, Renaissance, Romantic

### Electronic
Abstract, Acid, Acid House, Acid Jazz, Ambient, Bassline, Beatdown, Berlin-School, Big Beat, Bleep, Breakbeat, Breakcore, Breaks, Broken Beat, Chillwave, Chiptune, Dance-pop, Dark Ambient, Darkwave, Deep House, Deep Techno, Disco, Disco Polo, Donk, Downtempo, Drone, Drum n Bass, Dub, Dub Techno, Dubstep, Dungeon Synth, EBM, Electro, Electro House, Electroclash, Euro House, Euro-Disco, Eurobeat, Eurodance, Experimental, Freestyle, Future Jazz, Gabber, Garage House, Ghetto, Ghetto House, Glitch, Goa Trance, Grime, Halftime, Hands Up, Happy Hardcore, Hard House, Hard Techno, Hard Trance, Hardcore, Hardstyle, Hi NRG, Hip Hop, Hip-House, House, IDM, Illbient, Industrial, Italo House, Italo-Disco, Italodance, Jazzdance, Juke, Jumpstyle, Jungle, Latin, Leftfield, Makina, Minimal, Minimal Techno, Modern Classical, Musique Concrète, Neofolk, New Age, New Beat, New Wave, Noise, Nu-Disco, Power Electronics, Progressive Breaks, Progressive House, Progressive Trance, Psy-Trance, Rhythmic Noise, Schranz, Sound Collage, Speed Garage, Speedcore, Synth-pop, Synthwave, Tech House, Tech Trance, Techno, Trance, Tribal, Tribal House, Trip Hop, Tropical House, UK Garage, Vaporwave

### Folk, World, & Country
African, Bluegrass, Cajun, Canzone Napoletana, Catalan Music, Celtic, Country, Fado, Flamenco, Folk, Gospel, Highlife, Hillbilly, Hindustani, Honky Tonk, Indian Classical, Laïkó, Nordic, Pacific, Polka, Raï, Romani, Soukous, Séga, Volksmusik, Zouk, Éntekhno

### Funk / Soul
Afrobeat, Boogie, Contemporary R&B, Disco, Free Funk, Funk, Gospel, Neo Soul, New Jack Swing, P.Funk, Psychedelic, Rhythm & Blues, Soul, Swingbeat, UK Street Soul

### Hip Hop
Bass Music, Boom Bap, Bounce, Britcore, Cloud Rap, Conscious, Crunk, Cut-up/DJ, DJ Battle Tool, Electro, G-Funk, Gangsta, Grime, Hardcore Hip-Hop, Horrorcore, Instrumental, Jazzy Hip-Hop, Miami Bass, Pop Rap, Ragga HipHop, RnB/Swing, Screw, Thug Rap, Trap, Trip Hop, Turntablism

### Jazz
Afro-Cuban Jazz, Afrobeat, Avant-garde Jazz, Big Band, Bop, Bossa Nova, Contemporary Jazz, Cool Jazz, Dixieland, Easy Listening, Free Improvisation, Free Jazz, Fusion, Gypsy Jazz, Hard Bop, Jazz-Funk, Jazz-Rock, Latin Jazz, Modal, Post Bop, Ragtime, Smooth Jazz, Soul-Jazz, Space-Age, Swing

### Latin
Afro-Cuban, Baião, Batucada, Beguine, Bolero, Boogaloo, Bossanova, Cha-Cha, Charanga, Compas, Cubano, Cumbia, Descarga, Forró, Guaguancó, Guajira, Guaracha, MPB, Mambo, Mariachi, Merengue, Norteño, Nueva Cancion, Pachanga, Porro, Ranchera, Reggaeton, Rumba, Salsa, Samba, Son, Son Montuno, Tango, Tejano, Vallenato

### Non-Music
Audiobook, Comedy, Dialogue, Education, Field Recording, Interview, Monolog, Poetry, Political, Promotional, Radioplay, Religious, Spoken Word

### Pop
Ballad, Bollywood, Bubblegum, Chanson, City Pop, Europop, Indie Pop, J-pop, K-pop, Kayōkyoku, Light Music, Music Hall, Novelty, Parody, Schlager, Vocal

### Reggae
Calypso, Dancehall, Dub, Lovers Rock, Ragga, Reggae, Reggae-Pop, Rocksteady, Roots Reggae, Ska, Soca

### Rock
AOR, Acid Rock, Acoustic, Alternative Rock, Arena Rock, Art Rock, Atmospheric Black Metal, Avantgarde, Beat, Black Metal, Blues Rock, Brit Pop, Classic Rock, Coldwave, Country Rock, Crust, Death Metal, Deathcore, Deathrock, Depressive Black Metal, Doo Wop, Doom Metal, Dream Pop, Emo, Ethereal, Experimental, Folk Metal, Folk Rock, Funeral Doom Metal, Funk Metal, Garage Rock, Glam, Goregrind, Goth Rock, Gothic Metal, Grindcore, Grunge, Hard Rock, Hardcore, Heavy Metal, Indie Rock, Industrial, Krautrock, Lo-Fi, Lounge, Math Rock, Melodic Death Metal, Melodic Hardcore, Metalcore, Mod, Neofolk, New Wave, No Wave, Noise, Noisecore, Nu Metal, Oi, Parody, Pop Punk, Pop Rock, Pornogrind, Post Rock, Post-Hardcore, Post-Metal, Post-Punk, Power Metal, Power Pop, Power Violence, Prog Rock, Progressive Metal, Psychedelic Rock, Psychobilly, Pub Rock, Punk, Rock & Roll, Rockabilly, Shoegaze, Ska, Sludge Metal, Soft Rock, Southern Rock, Space Rock, Speed Metal, Stoner Rock, Surf, Symphonic Rock, Technical Death Metal, Thrash, Twist, Viking Metal, Yé-Yé

### Stage & Screen
Musical, Score, Soundtrack, Theme