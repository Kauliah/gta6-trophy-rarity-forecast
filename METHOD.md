# Method — Leonida Week-One Trophy Forecast

## The problem in one line

Rate_i(t) = 100 · E_i(t) / N(t). Both factors move fast in launch week, in
opposite directions for different trophies, so the rate itself is the wrong
object to extrapolate. The program forecasts E_i and N separately in
log-time and divides.

## Inputs (from the practice cases, the only real data available)

- 55 trophies: 1 platinum, 2 gold, 6 silver, 46 bronze.
- Practice histories are a single distinct snapshot: N = 638, rates frozen
  from 2026-11-18T23:50Z through the checkpoint. Earners are small integers
  (0.63% = 4/638, 0.78% = 5/638, 1.10% = 7/638, 2.04% = 13/638).
- The first refresh (04:30Z) took N to 3,868 and lowered every rate except
  the one at 100% — pure dilution would put each rate at 16.5% of its last
  value, so the truth sits between that floor and the last rate.

## Model

- **Dedupe:** consecutive reads with identical `(players_tracked, rates)`
  collapse to one observation.
- **Time:** x = ln(1 + t), t = hours since 2026-11-19T00:00:00Z, floored at
  0.5 h. Target x for the final case (2026-11-26T00:00Z) is ln(169) ≈ 5.13.
- **Fits:** recency-weighted least squares (24 h half-life) of log N and
  log E_i on x. E_i = rate_i · N / 100 from the page; a zero rate is
  carried as half an earner.
- **Shrinkage:** fitted log-E slope b is replaced by
  λ·b + (1−λ)·b_class with λ = n/(n+6), n = distinct reads. With one
  distinct read the class prior rules; with ~200 it is the data.
- **Class priors** (early-phase slope, late-phase slope, saturation cap %,
  base logit σ): platinum (0.30, 0.60, 2, 1.00); chapter_early (0.90, 1.70,
  85, 0.70); chapter_late (0.75, 1.50, 70, 0.80); final (0.35, 1.00, 25,
  0.95); side_mission (0.80, 1.35, 60, 0.75); side_final (0.55, 1.10, 30,
  0.85); collectible (0.50, 1.00, 35, 0.90); grind (0.45, 0.90, 25, 0.90);
  activity (0.70, 1.20, 55, 0.80). N prior slope (1.4 early, 0.9 late).
  The early/late blend is logistic in the age of the last distinct read,
  centred at 18 h. These priors are assumptions, not measured data:
  early on, tracked-player inflow swamps progression (every rate dilutes);
  days in, progression outruns inflow for story trophies.
- **Saturation damping:** before extrapolating, the earners slope is
  multiplied by max(0.15, 1 − last_rate/cap), then the line is re-anchored
  through the last observed earner count. (An earlier version damped after
  anchoring and collapsed the forecast — fixed and re-tested; see below.)
- **Spread:** σ² = base² + 0.18·days + resid_E² + resid_N² (+ 0.30² when
  fewer than 3 distinct reads; + (0.45·staleness/12h)² for a stale last
  snapshot; + 0.15²/0.10² for rates under 0.5%/1%), clipped to [0.70, 1.60].
  Quantiles are inv-logit(logit(median) + z_τ·σ), clamped to [0.02, 99.5].
- **The 100% trophy** (id 2, "Just Like Paradise", last rate 100%) is held
  at the 99.5% cap with σ = 0.30: the page printed it at 100% all launch
  night and the first refresh left it there.

## Practice sanity checks (measured, kit input, both cases)

- Format: 55 rows, ids 1–55, five finite non-decreasing quantiles in
  [0, 100], files 2,779 / 2,765 bytes (limit 65,536). Valid.
- Determinism: two runs byte-identical (`diff -r` clean).
- Runtime: ~0.05 s per run (limit 8 s).
- Directional check: medians below the last rate for **54/54** trophies
  (excluding the 100% one). Median ratio median/last = **0.42** (p00) and
  **0.40** (p01), range 0.20–0.56 — between the pure-dilution floor (0.165)
  and persistence (1.0), near the log-midpoint, as required. Example p00
  medians: platinum 0.16% (last 0.63%), Chapter 1 0.71% (last 1.25%),
  final mission 0.30% (last 1.10%), 250 Stunts 0.31% (last 0.63%).

## Synthetic long-history test (assumptions labelled)

A fake final-like case was built in /tmp (not shipped): 299 distinct reads
over ~2 days, N 638 → 14,916 on a power law, per-trophy saturating
progression curves, refresh plateaus, occasional null rates, checkpoint
2026-11-21T11:30Z, horizon 6,510 min. The program ran in 0.08 s and:

- platinum median lowest (0.06%), Chapter 1 among the highest;
- medians above the last rate for 53/54 trophies (rates rise over 4.5 days
  in this synthetic world, as progression outruns inflow);
- medians close to the synthetic truth where it was defined independently
  of the entry's own priors: final mission 8.5% vs 8.1% true, Chapter 3
  33.5% vs 35%, Chapter 4 27.2% vs 28%, 100% Completion 0.54% vs 0.64%;
  Chapter 1 41% vs 55% and Chapter 2 43% vs 45% sit inside their wide
  90% bands (8.7–83.6% and 9.3–84.5%). The synthetic truth is my own
  construction, so this validates machinery and ordering, not the priors.
- Edge cases: single-read history, empty history, and empty trophy list
  all produce valid output (no crash).

## Known limits

- With a flat history the forecast is entirely prior-driven; the priors
  above are reasoned, not fitted — the real final history (hundreds of
  distinct reads) is what lets the data take over.
- No external data is used: no network, no comparables, no scraped rates.
  If Exophase's refresh cadence or the tracked-population mix changes
  after the deadline cut, the spread (not the median) is what absorbs it.
