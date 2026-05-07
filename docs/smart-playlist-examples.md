# Smart Playlist Examples

Example `.nsp` files for Navidrome that make use of the custom tags written by the music tagger. Each playlist demonstrates a different combination of fields and explains the logic behind the thresholds chosen.

## Deployment

Place `.nsp` files in any folder inside your music library, or in the path set by `PlaylistsPath` in `navidrome.toml`. Navidrome detects them automatically on the next library scan. To make a playlist visible to all users, set its visibility to **Public** in the Playlists view after it appears.

For a refresher on what each field measures and how reliable it is, see the **Tag Field Reference** document.

---

## A Note on Thresholds

All score fields run from 0–100. The examples below use values that performed well during testing on a varied library, but your mileage will vary depending on your collection. If a playlist is coming up empty or pulling in unexpected tracks, the first thing to adjust is the threshold — loosen a `gt` value by 10–15 points or tighten a `lt` value in the same direction.

---

## The Playlists

---

### 1. Surf's Up
*Instrumental surf rock*

The simplest playlist in this collection and a good test that your custom tags are working. It combines a text match on the `mood` sub-genre field with an instrumental score filter. Because `Surf` as a label is relatively uncommon in the taxonomy, the instrumental filter does the heavy lifting on any borderline results — if a surf-adjacent track somehow made it through with vocals, a score above 60 on `instrumental` should catch it.

```json
{
  "name": "Surf's Up",
  "comment": "Instrumental surf rock — twangy guitars, spring reverb, no vocals needed.",
  "all": [
    { "contains": { "mood": "Surf" } },
    { "gt": { "instrumental": 60 } }
  ],
  "sort": "random",
  "limit": 50
}
```

**Tuning tips:**
- Lower the `instrumental` threshold to `45` if the playlist is too small — some surf tracks have faint backing vocals that push the score down.
- Swap `"Surf"` for `"Surf Rock"` if your library uses the longer label. Check a few tagged tracks first to confirm which string appears in the `mood` field.
- Add `{ "gt": { "mood_happy": 50 } }` to exclude any darker surf-adjacent material (surf-influenced post-punk, shoegaze, etc.) that might match on genre alone.

---

### 2. Late Night Wind Down
*Calm, melancholy, low energy*

Three conditions working together: the track must be relaxed, must not be aggressive, and should lean at least slightly sad. The sadness filter is intentionally loose — it exists mainly to exclude cheerful background music rather than to require genuine emotional weight. The result is a playlist that moves slowly and doesn't surprise you.

```json
{
  "name": "Late Night Wind Down",
  "comment": "Slow, calm, a little melancholy. For when the day is finally over.",
  "all": [
    { "gt": { "mood_relaxed": 72 } },
    { "lt": { "mood_aggressive": 18 } },
    { "gt": { "mood_sad": 40 } }
  ],
  "sort": "random",
  "limit": 100
}
```

**Tuning tips:**
- `mood_relaxed > 72` is a fairly tight filter. Drop it to `60` if you want more variety, especially if your library skews toward rock over ambient or classical.
- Adding `{ "gt": { "acoustic": 45 } }` shifts the result toward acoustic instruments, which suits this mood well.
- Adding `{ "lt": { "bpm": 100 } }` enforces a tempo ceiling if you find upbeat-but-calm tracks sneaking in.

---

### 3. Workout Fuel
*High intensity, high energy*

This playlist uses an `any` block to accept tracks from two different routes into high energy: either directly aggressive music, or fast danceable music that isn't necessarily harsh. The outer `all` block then requires that neither route produces something that is also calm — a track that somehow scores high on aggressive and relaxed simultaneously gets filtered out.

```json
{
  "name": "Workout Fuel",
  "comment": "Aggressive or fast and danceable. Nothing calm allowed.",
  "all": [
    {
      "any": [
        { "gt": { "mood_aggressive": 65 } },
        {
          "all": [
            { "gt": { "bpm": 140 } },
            { "gt": { "mood_dance": 62 } }
          ]
        }
      ]
    },
    { "lt": { "mood_relaxed": 30 } }
  ],
  "sort": "random",
  "limit": 75
}
```

**Tuning tips:**
- Raise `mood_aggressive` to `75` if you want only the most intense tracks and your library has a lot of hard rock that you find too moderate.
- The BPM + dance branch is what lets fast electronic and dance music in without requiring it to be harsh. Remove it if you want strictly aggressive music only.
- Add `{ "lt": { "mood_sad": 30 } }` to exclude brooding or heavy doom-adjacent tracks that score high on aggression but feel like the wrong energy.

---

### 4. Electronic Focus
*Instrumental electronic — study and concentration*

Instrumental electronic music occupies a specific niche that three fields triangulate well: high electronic character, high instrumental score (no prominent vocals), and low aggression (no harsh or intense content). This produces ambient, IDM, downtempo, and similar styles without pulling in aggressive electronic genres.

```json
{
  "name": "Electronic Focus",
  "comment": "Instrumental electronic for concentration. No vocals, nothing harsh.",
  "all": [
    { "gt": { "electronic": 68 } },
    { "gt": { "instrumental": 65 } },
    { "lt": { "mood_aggressive": 22 } }
  ],
  "sort": "random",
  "limit": 100
}
```

**Tuning tips:**
- The `electronic > 68` threshold is intentionally high because the electronic classifier can fire on dense or heavily produced music that is not genuinely electronic. If the playlist is too small, drop it to `55` but watch for false positives.
- Add `{ "gt": { "mood_relaxed": 45 } }` to further exclude anything too abrasive that snuck past the aggression filter.
- To allow minimal techno and similar genres that have a harder edge, relax `mood_aggressive` to `38`.

---

### 5. Friday Night
*High-energy dance and party music*

Both `mood_party` and `mood_dance` must be high — `party` alone can include exciting but hard-to-dance-to music, and `dance` alone can include slow-burn grooves. Together they target genuinely floor-filling tracks. The `mood_aggressive` ceiling is a guardrail against metal and punk that might score high on party energy but don't belong in the same mix.

```json
{
  "name": "Friday Night",
  "comment": "High party energy and genuinely danceable. The floor is yours.",
  "all": [
    { "gt": { "mood_party": 72 } },
    { "gt": { "mood_dance": 68 } },
    { "lt": { "mood_aggressive": 35 } }
  ],
  "sort": "-mood_dance,random",
  "limit": 100
}
```

**Tuning tips:**
- The sort `"-mood_dance,random"` floats the most danceable tracks to the top of the queue, then randomises within ties — so the playlist opens strong before becoming more varied.
- If your library is heavy on 70s disco or soul, those genres can score moderate on `mood_dance` but not hit 68. Drop the dance threshold to `55` to include them.
- Add `{ "gt": { "bpm": 110 } }` to enforce a minimum tempo if slow but danceable tracks (certain soul, R&B) are not what you want here.

---

### 6. Acoustic Sunday
*Relaxed acoustic music, with or without vocals*

A simpler playlist that uses only two fields: acoustic character and relaxation. The absence of an aggressive filter is intentional — acoustic music rarely scores high on aggression anyway, and omitting it keeps the JSON clean and the logic transparent.

```json
{
  "name": "Acoustic Sunday",
  "comment": "Acoustic instruments, relaxed pace. Good for a slow morning.",
  "all": [
    { "gt": { "acoustic": 65 } },
    { "gt": { "mood_relaxed": 58 } }
  ],
  "sort": "random",
  "limit": 80
}
```

**Tuning tips:**
- Add `{ "gt": { "instrumental": 55 } }` to prefer instrumental acoustic tracks — guitar pieces, classical, folk instrumentals.
- Add `{ "lt": { "bpm": 110 } }` to enforce a slow-to-mid tempo ceiling if upbeat folk or bluegrass feel out of place in the mix.
- To make this explicitly a singer-songwriter or vocal folk playlist, invert: `{ "lt": { "instrumental": 45 } }` to require significant vocal presence.

---

### 7. Genre Deep Dive: Metal
*Everything in the metal family*

The `mood` field shines for genre exploration because a single `contains` match catches every sub-genre label that includes the search term. `"Metal"` will match Heavy Metal, Speed Metal, Death Metal, Black Metal, Doom Metal, Thrash Metal, Folk Metal, Funeral Doom Metal, Goregrind, and every other label in the Rock taxonomy that contains the word — without having to list them individually.

```json
{
  "name": "Metal — All Subgenres",
  "comment": "Everything with Metal anywhere in the genre tag. Cast wide, sort by aggression.",
  "all": [
    { "contains": { "mood": "Metal" } }
  ],
  "sort": "-mood_aggressive,random",
  "limit": 200
}
```

This pattern generalises to any parent-genre exploration. Swap `"Metal"` for `"Jazz"`, `"Ambient"`, `"Punk"`, `"Reggae"`, and so on. The Appendix in the Tag Field Reference lists all 400 available labels.

**Variant — narrowed to a specific sub-genre:**

If you want only one sub-genre rather than the whole family, use `is` instead of `contains`. This requires an exact match, so check the label spelling in the Tag Field Reference taxonomy first.

```json
{
  "name": "Doom Metal",
  "comment": "Doom and Funeral Doom only — slow, heavy, monolithic.",
  "any": [
    { "is": { "mood": "Doom Metal" } },
    { "contains": { "mood": "Funeral Doom" } }
  ],
  "sort": "-mood_aggressive,random",
  "limit": 100
}
```

---

### 8. Discover: Happy but Unfamiliar
*Upbeat tracks you haven't played recently*

A practical discovery playlist that combines the mood score fields with Navidrome's built-in listening history. Requires high happy and party scores, then excludes anything played in the last 60 days — surfacing upbeat tracks that have been sitting unheard.

```json
{
  "name": "Discover: Happy but Forgotten",
  "comment": "Upbeat tracks you haven't played in the last 60 days.",
  "all": [
    { "gt": { "mood_happy": 72 } },
    { "gt": { "mood_party": 55 } },
    { "notInTheLast": { "lastPlayed": 60 } }
  ],
  "sort": "random",
  "limit": 50
}
```

**Tuning tips:**
- Increase `60` to `180` or `365` to surface genuinely forgotten music rather than just things you haven't queued recently.
- Add `{ "gt": { "playCount": 1 } }` to exclude tracks you have never played at all — keeping this as a "rediscovery" playlist rather than a general unfamiliar-music feed.
- Combine with a rating filter `{ "gt": { "rating": 2 } }` if you rate tracks, to exclude things you've heard and decided against.

---

## Combining Custom Tags with Built-in Fields

The custom tags are most powerful when paired with Navidrome's native fields. A few useful patterns:

**Mood-tagged music from a specific decade:**
```json
{
  "all": [
    { "gt": { "mood_relaxed": 65 } },
    { "inTheRange": { "year": [1970, 1979] } }
  ],
  "sort": "random",
  "limit": 60
}
```

**Loved tracks that are also genuinely danceable:**
```json
{
  "all": [
    { "is": { "loved": true } },
    { "gt": { "mood_dance": 65 } }
  ],
  "sort": "-mood_dance",
  "limit": 100
}
```

**Highly-rated instrumental music (no vocals, high rating):**
```json
{
  "all": [
    { "gt": { "rating": 3 } },
    { "gt": { "instrumental": 70 } }
  ],
  "sort": "-rating,random",
  "limit": 75
}
```