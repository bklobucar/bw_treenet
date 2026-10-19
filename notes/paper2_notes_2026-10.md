# Paper 2 notes — extending the canopy series past 2000 (October 2026)

Working notes for the change-detection paper (Paper 2). They record the decisions, evidence and open items from 14–15 October 2026. All cover figures are **preliminary** and are the share of canopy within the 2023 tätort study area (production masking), unless stated otherwise.

---

## 1. Summary

- The series now has three new eras (~2005, ~2010, ~2020) for Malmö, Gothenburg and Stockholm. All three were chosen to be leaf-on and are processed at 0.5 m with the same grayscale conversion used in training.
- The existing 1990s era was run at native 1 m. On the same imagery this gives **about 3.7–3.9 pp less canopy** than running at 0.5 m, the resolution v19 was trained at. The 1990s are therefore being rerun at 0.5 m (era label `1990r05`).
- Production converted RGB to grayscale with BT.601 weights, while the training tiles used a plain band mean. This is fixed, and no earlier v19 output was affected.
- The modern training imagery is now fully traced, and so are the labels. Labels for Gothenburg and Stockholm are Boverket laser tree height ≥ 3 m from leaf-off scans in 2019–2021; Malmö's are its own 2022 crown cover. All three ~2020 picks are independent of the training imagery.
- Ian's runs (six cities, mixed years) are useful for method checks: the noise floor between flights, the leaf-off effect and resolution sensitivity. They are not comparable enough to merge into the series.

---

## 2. Era design

### 2.1 Final flight picks

| Era | Malmö (12 sheets) | Gothenburg (17) | Stockholm (47) |
|---|---|---|---|
| ~2005 | 2004, `orto-historiska-2000-2004`, 0.5 m RGB, RT90 tiles | **2008**, `orto-goteborg-lerum-2008`, 0.25 m RGB | 2005, `orto-historiska-2005-2009`, 0.5 m RGB, RT90 tiles |
| ~2010 | 2010-06-16, `orto-a-2010`, 0.5 m | 2012-05-22/23, `orto-storgoteborg-2012`, 0.25 m (marginal) | 2011-06-04..29, `orto-o2-2011`, 0.5 m |
| ~2020 | 2020-06-02..08-01, `orto-a2-2020`, 0.16 m RGBI | 2022-06-14..24, `orto-g2-2022`, 0.16 m RGBI | 2019-06-18..07-26, `orto-o2-2019`, 0.16 m RGBI |

Comparison run: Gothenburg 2005 (`orto-historiska-2005-2009`, era label `2005cmp`). It looks leaf-off and is kept only to measure the leaf-off effect.

### 2.2 Leaf-on screening

The public Lantmäteriet STAC catalogue (`api.lantmateriet.se/stac-bild/v1`) was queried per 5 × 5 km sheet (script `32_leafon_scan.py`). Each flight was classified by its photo dates:

- **on:** 25 May – 15 Sep
- **marginal:** 10–24 May or 16–30 Sep
- **off:** anything else
- **unknown:** the catalogue gives the year only, as for most 2000–2009 collections

Main finding: **many post-2000 flights are leaf-off.**

| City | Leaf-off or marginal flights (not used) |
|---|---|
| Malmö | 2011 Lund (04-30), 2012 (05-01), 2014 (04-27), 2016 (05-09), 2018 (04-12), 2022 (04-16), 2024 (05-21, marginal) |
| Gothenburg | 2014 (04-15..20), 2016 (05-12, marginal), 2018 (05-09..20), 2020 (04-17..20), 2024 (05-18, marginal), 2026 (04-23) |
| Stockholm | 2013 (05-05..20), 2021 (04-08..18), 2023 (05-09..11) |

Malmö has no leaf-on flight between 2012 and 2018, so a mid-2010s era isn't possible across all three cities.

### 2.3 Greenness check for year-only flights

The green chromatic coordinate, GCC = G/(R+G+B), was computed on catalogue thumbnails over land pixels (`33_fetch_thumbs.py`, `34_greenness.py`). Flights with known dates serve as calibration.

| City | Leaf-off references | Leaf-on references | Year-only flight | Reading |
|---|---|---|---|---|
| Malmö | 2014 (Apr): 0.343 | 2010: 0.361 · 2020: 0.359 | 2004: 0.361 | leaf-on |
| Stockholm | 2021: 0.340 · 2023: 0.338 | 2011: 0.355 · 2019: 0.352 | 2005: 0.366 | leaf-on (likely late May) |
| Gothenburg | 2020: 0.344 · 2014: 0.352 | 2022: 0.352 | 2005: 0.353 · 2008: 0.356 · 2009: 0.351 | not separable |

Values are median GCC across tiles.

- In Gothenburg, camera and processing differences mask the seasonal signal (April 2014 = June 2022).
- **Visual check of Gothenburg:** the 2005 thumbnails are olive-brown with straw-coloured grass and look leaf-off. 2008 looks like 2012. The 90th-percentile GCC agrees: 2008 0.378 ≈ 2012 0.377, against 2005 0.368 and 2009 0.365.
- **Decision:** Gothenburg ~2005 = 2008. The model test (2005 vs 2008 on the same sheets) is the final check; see §7.

---

## 3. Imagery preprocessing protocol (all new eras)

Script `36_fetch_build_era.py`:

1. Download tiles per sheet with Geotorget Basic auth, then delete them after the sheet is built.
2. Warp to EPSG:3006 on the 5 × 5 km sheet grid with `gdalwarp -tr 0.5 0.5 -r near`. RT90 (EPSG:3021) collections are reprojected in the same step.
3. Grayscale = `floor((R+G+B)/3)`, uint8, nodata 0. This is identical to `07_prepare_tiles_all_cities.py`, which made the modern training tiles (GDAL `ReadAsArray` with buffer resampling = nearest, then band mean, then uint8 truncation). Pan tiles pass through.
4. Output: `data/raw/<city>/<era>_OF_gray_<city>/<sheet>_<flight year>.tif`, 10000 × 10000 px.

Build results:

| Era | Sheets | Raw download | Sheets > 5% nodata |
|---|---|---|---|
| 2005 | 76 | 34.8 GB | 1: Malmö 616_37_50, 10.8%, Öresund coast |
| 2005cmp | 17 | 6.6 GB | 0 |
| 2010 | 76 | 30.5 GB | 2: Malmö 616_36_05 and 616_37_50, 24.2% each, coast; still to check |
| 2020 | not yet built | ~219 GB (Malmö 33, Gothenburg 51, Stockholm 134) | — |

Processing note: an early build of the RT90-based sheets used incomplete tile sets because of a de-duplication bug in `32`. That build was fixed and every affected sheet was rebuilt (3–5 tiles per sheet).

---

## 4. Grayscale conversion: training vs production

| Step | Conversion |
|---|---|
| Training tiles (`03`, `07_prepare_tiles_all_cities.py`) | `((r + g + b) / 3.0).astype(np.uint8)`, i.e. band mean, truncated |
| Production `rgb_to_gray` (before 14 Oct 2026) | BT.601: 0.299 R + 0.587 G + 0.114 B |
| Production now | `np.floor((r + g + b) / 3.0)`, matching training |

All existing v19 production sources (1960s/70s/90s) are single-band (`is_rgb=False`), so **no published or earlier output used the BT.601 path**. Only RGB inputs are affected; the new eras are converted before production anyway.

---

## 5. Training data provenance and label definition

### 5.1 Modern training imagery (4,461 tiles)

| City | Tiles | Source flight(s) | Check |
|---|---|---|---|
| Malmö | 597 | `rgbi_latest`: 2024 (orto-a2-2024), plus 1 tile from 2022 | file names |
| Gothenburg | 1,112 | `gtb_2016_*`: orto-g2-2016 (2016-05-12, marginal) | tile prefix |
| Gothenburg | 1,595 | `gtb_*`: orto-g2-2024 (2024-05-18, marginal) | 20 tiles correlated with 2024 imagery: r = 0.953–0.991 |
| Stockholm | 1,157 | `rgb_latest_sth`: 2011 + 2017 | file names |

- The training config in `07` was out of date (one folder nested one level deeper, and a year filter no longer present). It's now annotated. The tiles themselves are correct and come from the right city (coordinates checked).
- The `gtb/rgbi_latest` raw folder on HomePC holds a **copy of Stockholm 2023** imagery. It wasn't used for training.

### 5.2 Labels

| City | Product | Canopy definition | Date |
|---|---|---|---|
| Gothenburg | Boverket/Lantmäteriet *Trädtäckning i tätorter* (`tradtackning_binar3m`) | tree height ≥ 3 m from airborne laser, 1 m | laser 2019-11-29 – 2020-03-21, leaf-off (`LOVFRI_PER = ja`); photo year 2018 |
| Stockholm | same | same | laser 2020-02-19 – 2021-04-14, leaf-off; photo year 2019 |
| Malmö | Malmö city `krontackning2022` | mapped crown cover | 2022 |

Implications for the paper:

- The model learned a **height-based canopy definition (≥ 3 m)** in two cities. Shrubs under 3 m are non-canopy, which fits the Paper 1 shrub finding.
- Leaf-off laser can under-detect the edges of deciduous crowns. State this as a known label bias.
- Image–label gaps in training: Stockholm 2011 vs 2020–21 (up to 10 years) and Gothenburg 2016 vs 2019–20. That's label noise from real change in between, which may bias the model against young or recently planted trees.

### 5.3 Independence of new eras from training

| Era | Malmö | Gothenburg | Stockholm |
|---|---|---|---|
| ~2005 | independent | independent | independent |
| ~2010 | independent (2010) | independent (2012) | **overlaps training** (2011) |
| ~2020 | independent (2020; training 2024) | independent (2022; training 2016/2024) | independent (2019; training 2011/2017) |

The labels stay **a calibration point, not an era**. They come from a different method (laser) and are what the model was trained on. Compare the ~2020 maps with them per city, inside and outside the training-tile footprint. Best match in time: Stockholm 2019 vs laser 2020–21, Malmö 2020 vs 2022, Gothenburg 2022 vs laser 2019–20.

---

## 6. Resolution effect: the 1990s

Production ran the 1990s at native 1 m. v19's training data is almost entirely 0.5 m: 4,461 modern tiles and the 1960s/70s historical tiles. `tiles_1990` holds only one 1 km tile per city, at 1 m.

**Test (`37_res_test_1990s.py`):** the same 4 sheets per city, production at 1 m vs the same imagery upsampled to 0.5 m (bilinear).

| City | 1 m | 0.5 m | Difference | Ratio |
|---|---|---|---|---|
| Malmö | 8.80% | 12.46% | +3.7 pp | 1.42 |
| Gothenburg | 13.50% | 17.22% | +3.7 pp | 1.28 |
| Stockholm | 17.39% | 21.25% | +3.9 pp | 1.22 |

Per sheet (valid sheets only), the 0.5 m to 1 m ratio was 1.22–1.59.

**Decision:** replace the 1990s with the 0.5 m run (`1990r05`). Keep the 1 m run as a sensitivity result.

- **Consequence:** the 1990s values in `canopy_change_v19.csv` (dated 2026-10-14) are biased low by about 4 pp. The reported 1970s → 1990s drop, including Malmö's "dip then rise", must be re-examined with `1990r05`.
- **Caveat:** upsampled 1 m film is still blurrier than native 0.5 m imagery, so some source-quality effect remains (see §8).

**Accuracy against 1990s labels:** the only 1990s labels (`clips_for_labelling/labels_*_1990`, one 1 km clip per city) are the same tiles used in training (`07_train_v18.py`, `25_train_finetune.py` include `tiles_1990`). An evaluation on them is not independent and favours 1 m.

Result (`38_eval_1990_labels.py`, 1 km clip per city, 0.5 m evaluation grid):

| City | Labelled canopy | Run | Predicted cover | IoU | Precision | Recall |
|---|---|---|---|---|---|---|
| Malmö | 13.5% | 1 m | 8.3% | 0.416 | 0.771 | 0.475 |
| Malmö | 13.5% | 0.5 m | 14.7% | **0.471** | 0.614 | 0.670 |
| Gothenburg | 4.6% | 1 m | 6.7% | 0.415 | 0.493 | 0.726 |
| Gothenburg | 4.6% | 0.5 m | 5.6% | **0.475** | 0.586 | 0.714 |
| Stockholm | 33.4% | both | 0.0% | 0 | 0 | 0 |

- **0.5 m scores higher in both evaluable cities**, even though these clips were in training at 1 m. At 1 m the model misses half the labelled canopy in Malmö (recall 0.475).
- **Masking caveat** (`39` diagnosis): production predicts only inside the 2023 tätort (prob = −1 elsewhere).
  - Malmö's clip is 100% inside the tätort, so its scores are valid.
  - Gothenburg's clip is only 25% inside, so the table above counts the masked 75% as missed canopy. A rerun restricted to predicted pixels is queued (`38`, updated).
  - Stockholm's clip is 0% inside, so it can't be evaluated from production maps; it would need an unmasked run.
- **Whole-city effect** (from `20_aggregate_canopy.py`): 1990r05 minus 1990 = Malmö +4.0, Gothenburg +4.2, Stockholm +4.3 pp. That matches the 4-sheet test.
- **To do:** label a few unseen 1990s windows per city for an independent check.

---

## 7. Preliminary results (v19, 0.5 m, % of tätort study area)

Source: `results/canopy_change_v19_ext.csv` (`20_aggregate_canopy.py`, 15 Oct 2026). Each era is measured on its own valid area, so the footprints differ (see the caveat below).

| City | 1960s | 1970s | 1990s (1 m, old) | **1990s (0.5 m)** | ~2005 | ~2010 |
|---|---|---|---|---|---|---|
| Malmö | 6.64 | 5.65 | 8.26 | **12.26** | 10.87 (2004) | 11.67 (2010) |
| Gothenburg | 19.45 | 21.23 | 16.08 | **20.31** | 31.03 (2008) | 28.72 (2012) |
| Stockholm | 27.83 | 31.03 | 21.35 | **25.60** | 36.90 (2005) | 35.32 (2011, training imagery) |

Frames and valid area: Malmö 12 frames, about 118 km², in all eras. Gothenburg 16 frames (190–207 km²) in the 1960s/70s, 17 frames (about 220 km²) from the 1990s. Stockholm 36 frames (417–472 km²) to the 1990s, **47 frames (524 km²) from ~2005**.

**Leaf-off effect, measured.** On the same 17 Gothenburg sheets, the 2005 flight (looks leaf-off) gives **22.65%** and the 2008 flight gives **31.03%**: −8.4 pp, about 5× the noise floor between flights. This validates the leaf-on screening and shows that one wrong flight can create or erase a decade of apparent change.

### 7.1 Common footprint (the series to use)

Source: `results/canopy_common_footprint.csv` (`39_common_footprint.py`, 15 Oct 2026). The study area per city is the 2023 tätort ∩ imagery valid in every series era.

| City | Common area | 1960s | 1970s | 1990s (0.5 m) | ~2005 | ~2010 |
|---|---|---|---|---|---|---|
| Malmö | 118.8 km² (12 sheets) | 6.60 | 5.50 | 12.23 | 10.87 | 11.64 |
| Gothenburg | 191.3 km² (16 sheets) | 19.31 | 21.04 | 20.44 | 30.74 | 28.42 |
| Stockholm | 372.0 km² (25 sheets) | 25.52 | 28.26 | 25.32 | 34.37 | 33.12 |

Checks on the same footprint:

| City | 1990s at 1 m (old) | 1990s at 0.5 m | Δ | ~2005 | ~2005 degraded to 1 m and back | Δ | Gothenburg 2005 flight (leaf-off) |
|---|---|---|---|---|---|---|---|
| Malmö | 8.23 | 12.23 | +4.0 | 10.87 | 10.58 | −0.3 | — |
| Gothenburg | 16.44 | 20.44 | +4.0 | 30.74 | 30.56 | −0.2 | 22.59 (−8.2 vs 2008) |
| Stockholm | 21.36 | 25.32 | +4.0 | 34.37 | 34.21 | −0.2 | — |

What this establishes:

1. **The footprint change doesn't explain the step.** On a fixed area, Gothenburg still rises +10.3 pp and Stockholm +9.1 pp from the 1990s to ~2005.
2. **Inference resolution matters a lot** (+4.0 pp in all three cities). **Source resolution does not:** ~2005 imagery degraded to 1 m and upsampled again loses only 0.2–0.3 pp, below the noise floor. So the 1990s → ~2005 step is **not** caused by the 1990s being 1 m imagery.
3. **What remains to explain the step:** (a) real canopy gain in Gothenburg and Stockholm between 1995 and 2005/2008; or (b) film vs digital radiometry. Panchromatic film has a different spectral response to an RGB band mean, plus grain, contrast and scanning effects.
   - Against a pure source effect: Malmö makes the same film → digital switch (1998 → 2004) with **no** step (12.2 → 10.9).
   - For one: Malmö's canopy is mostly street and garden trees, while Gothenburg and Stockholm include large forest patches whose texture may read differently on film.
4. **Decisive test, queued:** panchromatic film flights from just before the digital era. Stockholm has 1999 and 2000; Gothenburg has 2001; Malmö 2000 has partial coverage only. All are 1 m film, processed exactly like the 1990s (upsampled to 0.5 m).
   - Film 1995 ≈ 1999 ≈ 2000, then a jump to digital 2005 → **source effect**. Real canopy can't gain 9–10 pp in 4–5 years.
   - A gradual rise across 1995 → 1999 → 2000 → 2005 → **real change**.

Malmö's 1970s → 1990s rise (5.5 → 12.2, both film) is a separate open question.

---

## 8. Partner runs (Ian, `v19_UniSthm`)

54 GeoPackages: Gothenburg, Malmö, Södermalm, Stockholm, Umeå and Uppsala, 1995–2018, threshold 0.6. Summary table in `partner_summary.csv` (`35_summarise_partner.py`).

What they show:

- **Noise floor between flights:** Södermalm, 9 tiles at 0.25 m: 2009 19.5 · 2013 20.4 · 2015 22.3 · 2017 20.7%, i.e. **±1.5 pp**. Change between eras must exceed this.
- **Leaf-off depression:**
  - Gothenburg, 2 tiles: 2008 37.3 · 2012 30.5 · **2014 (April) 26.9** · 2016 35.0%.
  - Södermalm, 4 sheets: 2005 27.4 · **2008 20.6** · 2011 26.5%. Stockholm 2008 is year-only and probably leaf-off; not used.
- **Step at the change of source resolution:**
  - Umeå: 1 m years (1996–2003) mean 12.2% vs 0.5 m years (2007–2012) mean 16.7%, i.e. +4.5 pp.
  - Uppsala: 15.4% vs 19.8%, i.e. +4.4 pp.

Why the runs aren't merged:

- Ian reports resampling all imagery to 1 m. The smallest polygon in each file equals the native pixel size, though (0.0625 m² at 0.25 m, 0.0256 m² at 0.16 m). Either the predictions were resampled back to native before polygonising, or not every year ran at 1 m. **Unresolved.**
- If everything ran at 1 m, the absolute levels are biased low (§6). The Umeå/Uppsala step then reflects the source (film vs digital), not the inference resolution.
- Other issues: leaf-off years; training imagery among the years (Gothenburg 2016, Stockholm 2011 and 2017); unknown grayscale conversion; footprints that vary between years.
- **Use:** method checks only (noise floor, leaf-off effect, resolution sensitivity). Proposed a shared protocol: 0.5 m, leaf-on screening (`32`), band-mean grayscale, tätort mask, threshold 0.6. Under it, Umeå and Uppsala can join the series.

---

## 9. Open items

- [x] ~2005 cover values for all three cities (`results/canopy_change_v19_ext.csv`).
- [x] 1990s label evaluation, 1 m vs 0.5 m: 0.5 m better in Malmö and Gothenburg.
- [x] Stockholm 1990s clip scores zero: the clip lies outside the 2023 tätort, so production masks it.
- [x] Common-footprint series (`39`) and the degraded-2005 test: source resolution is not the cause of the step.
- [ ] Film-vs-digital test: pan 1999/2000/2001 flights (`pan1999`, `pan2000`) on their own footprint (`39 --own-footprint`).
- [ ] Gothenburg label evaluation restricted to predicted (tätort) pixels (`38`, updated).
- [ ] Unseen 1990s test windows per city for an independent accuracy check.
- [ ] Build the ~2020 era (~219 GB raw at 0.16 m): convert and delete per sheet; check the quota first.
- [ ] Add `2020` to `20_aggregate_canopy.py` once it's built.
- [ ] Check the Malmö 2010 coastal sheets (24.2% nodata).
- [ ] Compare the ~2020 maps with the labels, split by training footprint.
- [ ] Questions to Ian: script used; how the polygons were made; grayscale method; collections per year.
- [ ] Rerun change rasters and the USB/QGIS package with `1990r05` and the new eras.

---

## 10. Scripts and code changes

| File | What |
|---|---|
| `32_leafon_scan.py` | STAC per sheet, flight dates and leaf status, picks per era, `tiles_<era>.txt` (de-duplication bug fixed 15 Oct) |
| `33_fetch_thumbs.py` | catalogue thumbnails per flight |
| `34_greenness.py` | GCC per flight from thumbnails |
| `35_summarise_partner.py` | summary of partner GeoPackages (SQLite read, no GIS needed) |
| `36_fetch_build_era.py` | download, warp to 0.5 m, band-mean grayscale, delete raw |
| `37_res_test_1990s.py` | 1990s 1 m vs 0.5 m test copies and comparison (`compare` step superseded by production runs) |
| `38_eval_1990_labels.py` | both 1990s runs scored against the 1990s label clips |
| `39_common_footprint.py` | cover per era on one common footprint per city, plus a diagnosis of the label clips |
| `19_production_inference.py` | `rgb_to_gray` → band mean; `--era` choices read from `SOURCES`; new eras `1990r05`, `1990r10`, `2005`, `2005cmp`, `2010`, `2020` |
| `20_aggregate_canopy.py` | new eras added; writes `canopy_change_v19_ext.csv` (original csv untouched) |
| `07_prepare_tiles_all_cities.py` (HomePC) | config annotated with the true training sources |
## Update 19 Oct 2026: the film → digital step

### Bottom line

- The 1990s → ~2005 rise in Gothenburg (+10.3 pp) and Stockholm (+9.1 pp) is **a source effect, not real canopy gain**.
- The jump appears exactly where the imagery switches from panchromatic film to digital colour. The last film flights (1999–2001) match the 1990s.
- The step sits in **closed stands**: built-up windows barely move, densely wooded ones jump 20–30 pp, and up to half of the gained area is whole patches with no film-era canopy within 5 m. This matches Malmö showing no step, since its canopy is mostly street and garden trees.
- **Cause still open, and pending a visual check of the image strips:**
  - (A) the model under-calls dense stands on film; or
  - (B) the film flights are leaf-off. Lantmäteriet metadata gives only the year (`leaf=unknown`), and the same applies to the 1990s flights.

### 1. Film-vs-digital test

Pre-digital panchromatic film flights, processed exactly like the 1990s: 1 m film, upsampled to 0.5 m (bilinear), production v19. Source: `results/canopy_own_footprints.csv` (`39 --own-footprint`, job `film_analysis`).

Each row is measured on the area where that film flight has imagery (2023 tätort ∩ imagery valid in all series eras and the film flight).

| City, footprint | Area | 1960s | 1970s | 1990s (0.5 m) | **Late film** | Digital ~2005 | ~2010 |
|---|---|---|---|---|---|---|---|
| Gothenburg, 2001 flight | 191.3 km², 16 sheets | 19.31 | 21.04 | 20.44 | **19.41** (2001) | 30.74 (2008) | 28.42 |
| Stockholm, 1999 flight | 162.0 km², 15 sheets | 25.83 | 25.62 | 24.67 | **25.30** (1999) | 31.84 (2005) | 31.17 |
| Stockholm, 2000 flight | 210.0 km², 15 sheets | 25.28 | 30.30 | 25.83 | **22.42** (2000) | 36.32 (2005) | 34.63 |

Full study area, from production: Gothenburg 2001 19.44%, Stockholm 1999 25.75%, Stockholm 2000 23.95%.

- The film years (1995, 1999, 2000/01) stay within about ±1 pp of each other. The exception is Stockholm 2000 at −3.4 pp, which may be a partly leaf-off flight.
- The step comes at the switch to digital: **+11.3 pp** (Gothenburg), **+6.5 pp** (Stockholm 1999 footprint), **+13.9 pp** (Stockholm 2000 footprint). Real canopy can't gain 7–14 pp in 5–7 years and then stay flat to 2010.
- Already excluded (15 Oct, common footprint):
  - **Footprint change:** the step persists on a fixed area.
  - **Source resolution:** ~2005 degraded to 1 m and back loses only 0.2–0.3 pp.
- What remains is **film vs digital radiometry**: spectral response, grain, contrast, scanning. Possibly combined with leaf state.

Malmö has no full-coverage film flight around 2000 (partial in 2000: 9 of 12 sheets), so it isn't in this test. It crosses the film → digital switch with no step (1990s 12.2 → ~2005 10.9).

### 2. Where the step lands (`40_film_digital_windows.py`)

Pixels valid in both film and digital (inside tätort, not water), cut into 250 m windows. The step is broken down by how wooded each window is in 2010. Gained pixels are classed by distance from film-era canopy.

| | Gothenburg (2001 → 2008) | Stockholm (1999 → 2005) | Stockholm (2000 → 2005) |
|---|---|---|---|
| Area scored | 219.7 km², 17 sheets | 222.0 km², 27 sheets | 291.1 km², 27 sheets |
| Film → digital cover | 19.44 → 31.03 | 25.75 → 32.58 | 23.95 → 39.50 |
| Net step | +11.59 pp | +6.84 pp | +15.55 pp |
| Gross gain / loss | 15.17 / 3.57 pp | 11.19 / 4.35 pp | 18.73 / 3.18 pp |
| Gain ≤1 m from film canopy (wider outlines) | 16.2% | 21.2% | 15.0% |
| Gain 1–5 m (near existing crowns) | 39.3% | 44.6% | 35.6% |
| Gain >5 m (whole new patches) | **44.5%** | 34.2% | **49.4%** |

Step (pp) by 2010 cover of the window:

| 2010 cover of window | Gothenburg | Stockholm 1999 | Stockholm 2000 |
|---|---|---|---|
| 0–10% | +1.2 | +0.3 | +3.4 |
| 10–25% | +7.0 | +4.7 | +8.4 |
| 25–50% | +14.8 | +7.7 | +15.0 |
| 50–75% | +23.2 | +10.8 | +22.3 |
| 75–100% | **+31.6** | +9.0 | **+22.2** |
| Share of net step from windows ≥50% wooded | 39.5% | 25.1% | 38.3% |

In the extreme windows, film-era predictions call almost nothing where digital calls closed canopy. For example, Gothenburg 640_30_05 (E309250 N6403500) goes from 7% (2001) to 90% (2008), with 82% in 2010. Gothenburg's eight largest-step windows all go from 7–28% to 86–97%, with 2010 confirming 35–92%.

**Caveat:** binning by 2010 cover (digital) partly correlates with the digital-era values. The trend is robust, but the bin shares shouldn't be read as exact. The extreme windows are far beyond any binning effect.

Image strips: `results/film_windows_{gtb_pan2000,sth_pan1999,sth_pan2000}.png`. Columns are 1990s | film | digital ~2005 | 2010, with predicted canopy outlined in red. Some cells are blank because the flights don't cover those sheets (Stockholm 1999 and 2000 cover different sheets; the 1990s don't cover the eastern sheets).

### 3. Interpretation and implications

- **(A) Model failure on film in closed stands (leading hypothesis).**
  - The historical training tiles (102/142/172) likely contain few closed stands (to verify against the tile footprints).
  - On film, a dense stand may read as uniform dark texture rather than individual crowns.
  - Consistent with the urban label clips, which show no film under-calling (section 4).
- **(B) Leaf-off film flights.** Deciduous stands lose their texture.
  - Against it as the full story: 1995, 1999 and 2001 agree with each other, which would require all of them to be leaf-off to a similar degree.
  - Stockholm 2000 being lowest fits a partly leaf-off flight.
- **Deciding check:** the image strips.
  - Trees visible in the film column but not outlined → (A).
  - Grey, lacy crowns → (B).

**Consequences for Paper 2 (if A holds):**

1. No change can be reported across the film/digital break for Gothenburg and Stockholm. The series is two internally consistent sub-series: **film 1960–2001** and **digital 2005–2020**.
2. Absolute film-era cover is likely **biased low in wooded areas**: all film eras (1960s, 1970s, 1990s) in Gothenburg and Stockholm. Change *between* film eras is affected only where the forest share changed.
3. Malmö is least affected (few closed stands). This is consistent with no step.
4. **Fix:** historical training tiles from closed stands, as Gui advised (era-matched labels are the bottleneck). Candidate windows are Gothenburg 640_30_05 (E309250 N6403500) and 640_31_50 (E310250 N6405500). Labelling them in film imagery also gives a forest-stand accuracy figure for Paper 1, which the urban clips can't provide.

### 4. 1990s label evaluation (corrected)

Source: `results/eval_1990_labels.csv` (`38`, 19 Oct). Production predicted pixels only, i.e. inside the 2023 tätort.

| City (scored part of clip) | Run | Cover vs labels | IoU | Precision | Recall |
|---|---|---|---|---|---|
| Malmö (100%) | 1 m | 8.3 vs 13.5 | 0.416 | 0.771 | 0.475 |
| | 0.5 m | 14.7 vs 13.5 | **0.471** | 0.614 | 0.670 |
| Gothenburg (25%) | 1 m | 27.1 vs 18.4 | 0.415 | 0.493 | 0.726 |
| | 0.5 m | 22.4 vs 18.4 | **0.475** | 0.586 | 0.714 |
| Stockholm (0%) | — | clip lies entirely outside the 2023 tätort, so it can't be evaluated from production maps | | | |

- 0.5 m beats 1 m in both evaluable cities and brings cover closer to the labels. This supports switching the 1990s to 0.5 m on labels as well as on the series.
- Caveats:
  - The clips were in training (at 1 m), so these aren't independent accuracies.
  - Gothenburg's scored area is only 0.25 km².
  - Neither clip contains a closed stand.
- On urban canopy, film at 0.5 m does not under-call (+1.2 pp Malmö, +4.0 pp Gothenburg).

### 5. Code changes and bugs (19 Oct)

- **`38_eval_1990_labels.py`:**
  - The canopy maps carry `NoData Value=0`. Without `srcNodata="None"` the warp dropped every non-canopy pixel, scoring only predicted canopy (recall 1.000 by construction). Fixed.
  - The OGR layer is now held through its dataset (`lds = ogr.Open(shp); lyr = lds.GetLayer()`); before, it could be freed mid-use.
  - Scoring is restricted to pixels where production predicted (`prob >= 0`).
- **`39_common_footprint.py`:**
  - `--own-footprint` added (patched on the cluster via `patch_38_39.py`).
  - Its `warp` already used `srcNodata="None"`, so all `39` numbers are unaffected by the nodata issue.
  - Partial-coverage eras must go through `--own-footprint`, not `--extra`.
- **`40_film_digital_windows.py`** (new): step breakdown by distance from film canopy and by window woodedness, plus image strips.
- **Note:** the Arrhenius login node is x86 and the `gis` env is GH200-only, so GDAL commands need a GPU node.

### Open

- [x] Film-vs-digital test: the step is a source effect.
- [x] Spatial breakdown: the step sits in closed stands, much of it in whole patches.
- [x] 1990s label evaluation on predicted pixels (Gothenburg, Malmö).
- [ ] Inspect the image strips: model failure (A) vs leaf-off film (B).
- [ ] If A: label 1–2 closed-stand film windows, add them to training, retrain, and rerun the film eras and series.
- [ ] Decide how Paper 2 reports the series: two sub-series vs a calibrated single series.
- [ ] Unseen 1990s test windows per city (independent accuracy), including at least one closed stand.
- [ ] Malmö 1970s → 1990s rise (5.5 → 12.2, both film): separate open question.
