"""Leonida Week-One Trophy Forecast -- make.fun gta6-trophy-rarity-forecast-01.

Forecasts every GTA VI (PS5) trophy's Exophase earn rate at a case's target,
as five quantiles per trophy, from the house's reads of the trophy page.

Method
------
The page's numbers are two moving parts, not one: the earn rate of trophy i
is rate_i(t) = 100 * E_i(t) / N(t), where N(t) is the players Exophase tracks
and E_i(t) the earners of trophy i.  N grows fast in launch week (new tracked
players arrive constantly) while E_i grows only as players actually progress
through the game, so a rate can fall even while its earners multiply -- and
an early-story rate can rise even while N multiplies, because a far larger
share of the tracked population has finished Chapter 1 by day seven than by
hour three.  Forecasting the rate directly (persistence, or a straight line
in the rate) misses both effects, so this program decomposes, fits, and
recombines:

1. Dedupe.  Exophase refreshes on its own schedule, not the recorder's, so
   the page sits unchanged for hours between refreshes.  Consecutive reads
   with identical (players_tracked, rates) are one observation, kept once;
   otherwise a stale plateau would dominate every fit as if it were many
   independent reads.

2. Log-time fits.  With t the hours since release (2026-11-19T00:00:00Z,
   floored at 0.5 h so the pre-release reads still have a finite log-time),
   x = ln(1 + t).  Both log N and log E_i are fitted by recency-weighted
   least squares against x (24 h half-life), because launch-week growth is
   roughly power-law in time since release, i.e. linear in log-log space.
   E_i is taken from the page as rate_i * N / 100 (rates are printed to
   2 decimals, so E_i is effectively a small integer count early on); a
   zero rate is carried as half an earner so its log stays finite.

3. Shrinkage toward class priors.  Each trophy is classified from its
   name, description and tier into a progression class (platinum, early
   chapter, late chapter, final mission, side mission, side-mission
   finale, collectible, grind, activity).  A trophy's fitted log-E slope
   is shrunk toward its class's prior slope, with weight growing in the
   number of distinct reads: with one distinct read (the practice cases)
   the prior rules entirely; with ~200 distinct reads (the final case)
   the data rules.  The class prior slope also depends on launch phase:
   in the first hours the tracked population floods in faster than
   anyone can earn anything, so earners grow slower than N and every
   rate dilutes; days in, inflow slows while progression continues, so
   early-chapter earners outgrow N and their rates climb.  The prior
   blends between those two regimes on the age of the last distinct
   read (logistic, centred at 18 h).  N's prior slope is 1.4 early,
   0.9 later, on the same blend.

4. Recombine and cap.  rate = 100 * E_pred / N_pred at the target's
   x, capped by the class's saturation ceiling (a week-one platinum
   cannot plausibly sit at 5%; an early chapter cannot exceed the low
   80s) and floored at 0.02%.  The fitted earners slope is damped by
   (1 - last_rate / cap) before extrapolating, because a progression
   curve flattens as it saturates -- a straight log-log line would
   otherwise carry a fast early rise far past any plausible ceiling.
   A trophy last seen at 100% is held at the 99.5% cap: the page
   printed it at 100% through the launch night and the one refresh
   left it there.

5. Honest spreads.  Quantiles are the median on the logit scale plus
   z_tau * sigma, with sigma built from a per-class base (wider for
   rare, late and grind trophies), a horizon term (0.18 * days, so the
   4.5-day final extrapolation is far wider than the 90-minute practice
   one), the fits' residual scatter, and an extra term when the history
   carries no trend at all (a single distinct read).  Total sigma lands
   near 0.85 on the practice cases and 1.0-1.5 on the final horizon:
   the house baseline's 0.5*sqrt(days) is far too narrow for a refresh
   it has not seen, let alone a 4.5-day extrapolation.

Edge cases: a trophy with null rates in some reads uses its last
non-null rate; a case with a single read, or with no history at all,
still writes a valid forecast (priors + tier defaults); a missing
trophy list yields an empty forecast.  No network, no randomness:
two runs on the same case write byte-identical output.

Standard library only.  Run locally:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- class priors -----------------------------------------------------------
# Each class: (early-phase log-E slope, late-phase log-E slope,
#              saturation cap in percent, base logit sigma).
# Slopes are d log E / d log(1 + hours-since-release).  Early phase = the
# first hours, when tracked-player inflow swamps progression; late phase =
# days in, when progression outruns inflow for story trophies.
CLASS_PRIOR = {
    "platinum":      (0.30, 0.60, 2.0, 1.00),
    "chapter_early": (0.90, 1.70, 85.0, 0.70),
    "chapter_late":  (0.75, 1.50, 70.0, 0.80),
    "final":         (0.35, 1.00, 25.0, 0.95),
    "side_mission":  (0.80, 1.35, 60.0, 0.75),
    "side_final":    (0.55, 1.10, 30.0, 0.85),
    "collectible":   (0.50, 1.00, 35.0, 0.90),
    "grind":         (0.45, 0.90, 25.0, 0.90),
    "activity":      (0.70, 1.20, 55.0, 0.80),
}

# Prior slope for log N, (early, late).
N_PRIOR = (1.4, 0.9)
# Age (hours since release) at which the prior blend is 50/50, and its width.
PHASE_CENTRE = 18.0
PHASE_WIDTH = 6.0

HALFLIFE_HOURS = 24.0   # recency weighting of the log-time fits
SHRINK_K = 6.0          # distinct reads needed for the fit to half-dominate
MIN_T_HOURS = 0.5       # floor on hours-since-release inside log(1 + t)
RATE_FLOOR = 0.02
RATE_CAP = 99.5
SIGMA_MIN = 0.70
SIGMA_MAX = 1.60


def parse_ts(text):
    """ISO-8601 UTC timestamp -> aware datetime; None when unparseable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_since_release(when):
    if when is None:
        return 0.0
    return max((when - RELEASE).total_seconds() / 3600.0, 0.0)


def log_time(when):
    """x = ln(1 + t), t hours since release floored at MIN_T_HOURS."""
    return math.log1p(max(hours_since_release(when), MIN_T_HOURS))


def phase_weight(age_hours):
    """0 = fully late-phase prior, 1 = fully early-phase prior."""
    z = (age_hours - PHASE_CENTRE) / PHASE_WIDTH
    if z > 30:
        return 0.0
    if z < -30:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def classify(trophy):
    """Progression class from name / description / tier."""
    tier = (trophy.get("tier") or "").lower()
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    if tier == "platinum":
        return "platinum"
    if "final mission for" in desc:
        return "side_final"
    if "100% completion" in desc or "complete the final mission" in desc:
        return "final"
    if "complete chapter 1" in desc or "complete chapter 2" in desc:
        return "chapter_early"
    if "complete chapter" in desc:
        return "chapter_late"
    collectible_marks = ("collectible", "every spot", "every location",
                         "100 locations", "100 vehicles", "explore all",
                         "sets", "meditate")
    if any(m in text for m in collectible_marks):
        return "collectible"
    grind_marks = ("250", "20 different", "50 challenges", "earn $",
                   "spend $", "purchase all", "register 10",
                   "gold medal in all", "win 20", "complete 250")
    if any(m in text for m in grind_marks):
        return "grind"
    if desc.startswith("complete ") and "complete" in desc:
        return "side_mission"
    return "activity"


def dedupe(history):
    """Keep the first read of each run of identical snapshots.

    A snapshot is (players_tracked, rates tuple).  Exophase refreshes are
    step-like, so a plateau of identical reads is one observation.
    """
    out = []
    prev_key = None
    for read in history or []:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        out.append(read)
    return out


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, resid_sd)."""
    total = sum(ws)
    if total <= 0 or not xs:
        return 0.0, 0.0, 0.0
    xm = sum(w * x for x, w in zip(xs, ws)) / total
    ym = sum(w * y for y, w in zip(ys, ws)) / total
    sxx = sum(w * (x - xm) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - xm) * (y - ym) for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    intercept = ym - slope * xm
    if len(xs) >= 3 and total > 0:
        var = sum(w * (y - (intercept + slope * x)) ** 2
                  for x, y, w in zip(xs, ys, ws)) / total
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = 0.0
    return intercept, slope, resid


def clamp_rate(pct):
    return min(RATE_CAP, max(RATE_FLOOR, pct))


def logit(pct):
    p = min(99.99, max(0.01, pct)) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg > 30:
        return 100.0
    if lg < -30:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as a plain JSON number."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = dedupe(case.get("history") or [])
    horizon_min = float(case.get("horizon_minutes") or 0)
    days = max(horizon_min / 1440.0, 0.0)

    target_dt = parse_ts(case.get("target"))
    if target_dt is None:
        at_dt = parse_ts(case.get("at"))
        target_dt = at_dt
    x_target = log_time(target_dt)

    # --- last known state per trophy -------------------------------------
    last_rates = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                rate = float(rates[i])
                break
        last_rates.append(rate)

    # --- time grid of the distinct reads ---------------------------------
    times = [parse_ts(r.get("at")) or parse_ts(r.get("slot")) for r in history]
    xs = [log_time(t) for t in times]
    last_age = hours_since_release(times[-1]) if times else 0.0
    # How stale is the last distinct snapshot at the checkpoint?
    at_dt = parse_ts(case.get("at"))
    staleness_h = 0.0
    if at_dt is not None and times:
        staleness_h = max((at_dt - times[-1]).total_seconds() / 3600.0, 0.0)
    n_distinct = len(history)
    early_w = phase_weight(last_age)

    # Recency weights: 24 h half-life back from the last distinct read.
    ws = []
    for t in times:
        if t is None or times[-1] is None:
            ws.append(1.0)
        else:
            age = max((times[-1] - t).total_seconds() / 3600.0, 0.0)
            ws.append(0.5 ** (age / HALFLIFE_HOURS))

    # --- fit log N ---------------------------------------------------------
    n_slope_prior = N_PRIOR[0] * early_w + N_PRIOR[1] * (1.0 - early_w)
    n_vals = [r.get("players_tracked") for r in history]
    if n_distinct >= 2 and all(isinstance(v, (int, float)) and v > 0 for v in n_vals):
        ys = [math.log(float(v)) for v in n_vals]
        a_n, b_n, resid_n = wls(xs, ys, ws)
        lam = n_distinct / (n_distinct + SHRINK_K)
        b_n = lam * b_n + (1.0 - lam) * n_slope_prior
        # Re-anchor the intercept at the last read so the level is exact.
        a_n = math.log(float(n_vals[-1])) - b_n * xs[-1]
    elif n_distinct >= 1 and isinstance(n_vals[-1], (int, float)) and n_vals[-1] > 0:
        b_n = n_slope_prior
        a_n = math.log(float(n_vals[-1])) - b_n * xs[-1]
        resid_n = 0.0
    else:
        b_n, a_n, resid_n = n_slope_prior, math.log(638.0), 0.0
    b_n = max(0.0, min(2.5, b_n))
    n_pred = math.exp(a_n + b_n * x_target)
    n_pred = max(n_pred, 1.0)

    # --- per-trophy fits ---------------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        cls = classify(trophy)
        b_early, b_late, cap, base_sigma = CLASS_PRIOR[cls]
        b_prior = b_early * early_w + b_late * (1.0 - early_w)
        last = last_rates[i]

        # The 100% trophy: the page held it at 100% through launch night
        # and through the first refresh; hold it at the cap.  (99.9% or
        # above is the same saturated trophy, not a rate in motion.)
        if last is not None and last >= 99.9:
            med = RATE_CAP
            sigma = 0.30
            rows.append((trophy["id"], med, sigma))
            continue

        # Earners per distinct read, from rate * N / 100.
        pts_x, pts_y, pts_w = [], [], []
        for j, read in enumerate(history):
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            n_j = read.get("players_tracked")
            if not isinstance(n_j, (int, float)) or n_j <= 0:
                continue
            e = float(rates[i]) * float(n_j) / 100.0
            pts_x.append(xs[j])
            pts_y.append(math.log(max(e, 0.5)))
            pts_w.append(ws[j])
        if len(pts_x) >= 2:
            _a, b_e, resid_e = wls(pts_x, pts_y, pts_w)
            lam = len(pts_x) / (len(pts_x) + SHRINK_K)
            b_e = lam * b_e + (1.0 - lam) * b_prior
            # Anchor point: the last observed earner count.
            log_e_anchor, x_anchor = pts_y[-1], pts_x[-1]
        elif len(pts_x) == 1:
            b_e = b_prior
            log_e_anchor, x_anchor = pts_y[0], pts_x[0]
            resid_e = 0.0
        else:
            # Never observed: start from half an earner at the last x.
            b_e = b_prior
            log_e_anchor = math.log(0.5)
            x_anchor = xs[-1] if xs else 0.0
            resid_e = 0.0
        b_e = max(-0.5, min(3.0, b_e))
        # Saturation damping: a rate already close to its class ceiling
        # cannot keep growing on its fitted log-log slope -- progression
        # curves flatten as they saturate (logistic, not power-law).
        # At launch-night rates (~1%) this factor is ~1, so the practice
        # forecasts are untouched; near the ceiling it halves the slope.
        # Damping happens BEFORE re-anchoring, so the forecast line
        # still passes exactly through the last observed earner count.
        if last is not None and last > 0:
            b_e *= max(0.15, 1.0 - last / cap)
        a_e = log_e_anchor - b_e * x_anchor
        e_pred = math.exp(a_e + b_e * x_target)

        rate = 100.0 * e_pred / n_pred
        # Saturation cap for the class; never let a fitted explosion run.
        rate = min(rate, cap)
        if last is not None and last > 0:
            # A week-one rate cannot plausibly be more than ~40x its
            # launch-night print for a trophy already observed.
            rate = min(rate, last * 40.0)
        med = clamp_rate(rate)

        # --- spread on the logit scale ----------------------------------
        var = base_sigma * base_sigma
        var += 0.18 * days                    # horizon uncertainty
        var += min(resid_e, 0.5) ** 2         # fit scatter (earners)
        var += min(resid_n, 0.5) ** 2         # fit scatter (tracked)
        if n_distinct < 3:
            var += 0.30 ** 2                  # no trend visible at all
        if staleness_h > 0:
            stale = min(staleness_h / 12.0, 1.0)
            var += (0.45 * stale) ** 2         # refresh catch-up risk
        if last is not None and last < 0.5:
            var += 0.15 ** 2                  # rare trophies are noisier
        elif last is not None and last < 1.0:
            var += 0.10 ** 2
        sigma = math.sqrt(var)
        sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
        rows.append((trophy["id"], med, sigma))

    out_rows = []
    for tid, med, sigma in rows:
        lg = logit(med)
        qs = []
        for z in Z:
            q = inv_logit(lg + z * sigma)
            qs.append(round4(min(RATE_CAP, max(RATE_FLOOR, q))))
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        out_rows.append({"id": tid, "q": qs})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(INPUT_DIR.glob("*.in")) if INPUT_DIR.is_dir() else []
    for path in inputs:
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never crash a case: fall back to a valid flat forecast.
            rows = []
            for trophy in case.get("trophies") or []:
                rows.append({"id": trophy["id"],
                             "q": [0.02, 0.1, 0.5, 2.0, 10.0]})
            forecast = {"trophies": rows}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
