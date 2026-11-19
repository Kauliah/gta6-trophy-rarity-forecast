[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kauliah/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kauliah/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# Leonida Week-One Trophy Forecast

Forecasts every Grand Theft Auto VI (PlayStation 5) trophy's earn rate on
[Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — one week after release — for the make.fun bounty
`gta6-trophy-rarity-forecast-01`.

## Method

The page's numbers are two moving parts, not one: a trophy's earn rate is
`100 * earners / players_tracked`. The tracked population floods in during
launch week while earners grow only as players actually progress, so a rate
can fall while its earners multiply — and an early-story rate can rise even
while the tracked population multiplies, because a far larger share of
players has finished Chapter 1 by day seven than by hour three. `main.py`
therefore decomposes, fits, and recombines:

1. **Dedupe.** Exophase refreshes on its own schedule, so the page sits
   unchanged for hours between refreshes. Identical consecutive reads are
   one observation, kept once; otherwise a stale plateau would dominate
   every fit.
2. **Log-time fits.** With `x = ln(1 + hours since release)`, both `log N`
   and `log E_i` are fitted by recency-weighted least squares (24 h
   half-life). Launch-week growth is roughly power-law in time since
   release, i.e. linear in log-log space.
3. **Shrinkage toward class priors.** Each trophy is classified from its
   name, description and tier (platinum, early/late chapter, final
   mission, side mission, side-mission finale, collectible, grind,
   activity). Fitted earners slopes shrink toward the class prior, with
   the data's weight growing in the number of distinct reads. The prior
   itself blends between a launch phase (tracked inflow swamps
   progression, every rate dilutes) and a later phase (progression
   outruns inflow for story trophies).
4. **Saturation damping.** The fitted slope is damped by
   `(1 - last_rate / class_cap)` before extrapolating, because a
   progression curve flattens as it saturates; a straight log-log line
   would carry a fast early rise past any plausible ceiling.
5. **Honest spreads.** Quantiles are the median on the logit scale plus
   `z_tau * sigma`, with sigma from a per-class base, a horizon term
   (`0.18 * days`), the fits' residual scatter, and extra width when the
   history shows no trend at all. Total sigma is ~0.85 on the 90-minute
   practice cases and 1.0–1.5 on the 4.5-day final horizon — far wider
   than the house baseline's `0.5*sqrt(days)`, which the first refresh
   already blows through.

See `METHOD.md` for the derivation and the practice sanity-check numbers.

## Run

Python 3.13, standard library only, no network, no randomness — two runs
on the same case write byte-identical output.

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`, one row
per trophy, five quantiles in percent, non-decreasing, each file ≤ 65,536
bytes. Each case runs in well under the 8-second limit.

## Files

- `main.py` — the entrypoint (the whole model, single file).
- `METHOD.md` — model derivation and practice sanity-check numbers.
- `LICENSE` — MIT.
