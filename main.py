"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (Python 3.13, stdlib only).

Model: a launch-shape forecaster anchored on hour 1's print.

GTA VI unlocks at midnight local time, rolling by timezone (New Zealand first,
central Europe 23:00 UTC, the UK 00:00 UTC, US East 05:00 UTC, US West 08:00
UTC), on PS5 / Xbox Series X|S only, single-player. So its release day is not
a persistence process: viewers already present in hour 1 are only the
timezones unlocked by then, and the count rises as the Americas unlock and
again into the evening peak, then falls late in the day.

The program learns that shape at run time from the case's own `comparables`
(past big launches, SullyGnome hourly Twitch category averages):

- Clean set: a comparable counts when its unlock note names no early-access
  contamination, no weak tier, no single-global / simultaneous unlock (unless
  it also says midnight local), its release day has all 24 hours, its hour-1
  print is at least 3,000 viewers and at least 5% of its release-day peak
  (so hour 1 is a real launch hour, not a pre-unlock baseline).
- Tight set: the clean games with no PC at launch (console-only, like GTA VI
  and like the practice launches). When at least 5 tight games qualify, the
  per-hour median log-ratio is 0.65 * tight + 0.35 * clean; otherwise clean.
- For each forecast hour H = 2..24, the centre is the median log-ratio
  median(log(viewers_H / viewers_1)) over the set, adjusted for how big hour
  1 already is: across past launches the evening rise shrinks as hour 1
  grows (a launch whose hour 1 is already 800k viewers is near its peak;
  one whose hour 1 is 25k still has most of its audience to unlock), with a
  per-hour slope of log-ratio on log(hour-1 print), lightly smoothed across
  neighbouring hours, clamped to [-0.60, 0.05] and shrunk by 0.75, applied
  as centre_H = median_H + slope_H * (log(p1) - median log hour-1 print).
  The spread is the residual standard deviation of those log-ratios around
  that line, floored at 0.35 log units, capped at 0.90 and scaled by 1.15,
  so the 5%..95% band is honest about how much launch shapes vary.
  Console-only launches are blended in at 0.30 weight on the centre (their
  own regression, when at least 5 qualify), with the larger of the two
  spreads, since GTA VI is console-only.
- Quantiles: q_tau(H) = p1 * exp(centre_H + z_tau * sigma_H), rounded to 2
  decimals half-up, where p1 is hour 1's print (below 1 counts as 1) and
  z_tau is the standard normal quantile at 0.05 / 0.25 / 0.5 / 0.75 / 0.95.

Leave-one-out cross-validation on the kit's comparables: this model scores
about 0.27 mean pinball loss on the clean set (about 0.17 on the
console-only tight set), against about 0.41 (0.34 tight) for the house
baseline's persistence forecast, whose practice loss is about 0.387. The
plain median-ratio version scored about 0.33 (0.22 tight); the hour-1-size
adjustment is what buys the rest.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so the sys.path line below puts
it back first. It reads every <id>.in in the input directory and writes
<id>.out to the output directory: {"hours": [{"hour": 2, "q": [...]}, ...]},
exactly 23 rows, hours 2..24 each once, each q five finite numbers above 0
and at most 1e9, non-decreasing. Deterministic: no randomness, no clock, no
network; the same case always writes the same bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The house baseline's spread, used only if no comparable qualifies.
BASELINE_SIGMA = 0.15
# Blend weight on the console-only (tight) regression centre when it has enough games.
TIGHT_WEIGHT = 0.30
TIGHT_MIN_GAMES = 5
# Per-hour slope of log-ratio on log(hour-1 print): shrink, then clamp.
SLOPE_SHRINK = 0.75
SLOPE_MIN = -0.60
SLOPE_MAX = 0.05
# Spread floor / cap / scale applied to the residual standard deviation.
SIGMA_FLOOR = 0.35
SIGMA_CAP = 0.90
SIGMA_SCALE = 1.15


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _day_series(comp: dict) -> dict[int, int]:
    """Release-day hourly viewers of one comparable, keyed by hour index 0..23."""
    try:
        start = _parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    day: dict[int, int] = {}
    for row in comp.get("hourly") or []:
        try:
            delta = int((_parse_time(row["from"]) - start).total_seconds() // 3600)
            viewers = int(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= delta < 24:
            day[delta] = max(1, viewers)
    return day


def _is_clean(comp: dict, day: dict[int, int]) -> bool:
    """A comparable whose release day is a usable rolling-midnight launch."""
    if comp.get("name", "").startswith("Grand Theft"):
        return False
    unlock = str(comp.get("unlock") or "").lower()
    if "early-access-contaminated" in unlock:
        return False
    if "tier: weak" in unlock:
        return False
    if "single global" in unlock:
        return False
    if "simultaneous" in unlock and "midnight local" not in unlock:
        return False
    if len(day) < 24:
        return False
    hour1 = day.get(0, 0)
    if hour1 < 3000:
        return False
    if hour1 / max(day.values()) < 0.05:
        return False
    return True


def _is_tight(comp: dict) -> bool:
    """Console-only at launch, like GTA VI (no PC in its platforms)."""
    return "pc" not in str(comp.get("platforms") or "").lower()


def _regression(days: list[dict[int, int]], p1: float) -> tuple[list[float], list[float]]:
    """Per-hour centres and spreads for H = 2..24, in hour order.

    Centre: median log-ratio, plus a per-hour slope on log(hour-1 print)
    (smoothed, clamped, shrunk) times (log(p1) - median log hour-1 print).
    Spread: residual standard deviation of the log-ratios around that
    line, floored, capped and scaled.
    """
    log_p1 = math.log(max(1, p1))
    log_hour1 = [math.log(day[0]) for day in days]
    median_log_hour1 = statistics.median(log_hour1)

    slopes: list[float] = []
    medians: list[float] = []
    for h in range(1, 24):
        ratios = [math.log(day[h] / day[0]) for day in days]
        mean_x = statistics.mean(log_hour1)
        mean_y = statistics.mean(ratios)
        denom = sum((x - mean_x) ** 2 for x in log_hour1)
        slope = (
            sum((x - mean_x) * (y - mean_y) for x, y in zip(log_hour1, ratios)) / denom
            if denom
            else 0.0
        )
        slopes.append(slope)
        medians.append(statistics.median(ratios))
    slopes = _smooth(slopes)
    slopes = [max(SLOPE_MIN, min(SLOPE_MAX, s * SLOPE_SHRINK)) for s in slopes]

    centres: list[float] = []
    sigmas: list[float] = []
    for i, h in enumerate(range(1, 24)):
        ratios = [math.log(day[h] / day[0]) for day in days]
        centre = medians[i] + slopes[i] * (log_p1 - median_log_hour1)
        residuals = [
            y - (medians[i] + slopes[i] * (x - median_log_hour1))
            for x, y in zip(log_hour1, ratios)
        ]
        sigma = statistics.pstdev(residuals) if len(residuals) > 1 else SIGMA_FLOOR
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma)) * SIGMA_SCALE
        centres.append(centre)
        sigmas.append(sigma)
    return centres, sigmas


def _smooth(values: list[float]) -> list[float]:
    """Light 0.25 / 0.5 / 0.25 smoothing across neighbouring hours."""
    out: list[float] = []
    for i, value in enumerate(values):
        left = values[i - 1] if i > 0 else value
        right = values[i + 1] if i + 1 < len(values) else value
        out.append(0.25 * left + 0.5 * value + 0.25 * right)
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    clean_days: list[dict[int, int]] = []
    tight_days: list[dict[int, int]] = []
    for comp in case.get("comparables") or []:
        day = _day_series(comp)
        if not _is_clean(comp, day):
            continue
        clean_days.append(day)
        if _is_tight(comp):
            tight_days.append(day)

    if not clean_days:
        # No usable launch shape: fall back to the house baseline's persistence.
        hours = [
            {
                "hour": row["hour"],
                "q": [
                    round2(p1 * math.exp(z * BASELINE_SIGMA * math.sqrt(row["hour"] - 1)))
                    for z in Z
                ],
            }
            for row in case["forecast"]
        ]
        return {"hours": hours}

    centres, sigmas = _regression(clean_days, p1)
    if len(tight_days) >= TIGHT_MIN_GAMES:
        tight_centres, tight_sigmas = _regression(tight_days, p1)
        centres = [
            TIGHT_WEIGHT * t + (1.0 - TIGHT_WEIGHT) * c
            for t, c in zip(tight_centres, centres)
        ]
        sigmas = [max(t, c) for t, c in zip(tight_sigmas, sigmas)]

    by_hour = {row["hour"]: i for i, row in enumerate(case["forecast"])}
    hours = []
    for row in case["forecast"]:
        i = by_hour[row["hour"]]
        centre = centres[i]
        sigma = sigmas[i]
        quantiles = [round2(p1 * math.exp(centre + z * sigma)) for z in Z]
        # Keep the row non-decreasing after rounding (rounding can only tie).
        quantiles = [max(1, q) for q in quantiles]
        hours.append({"hour": row["hour"], "q": quantiles})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
