"""GTA VI trophy rarity forecast (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, five quantiles (5/25/50/75/95%)
of its earn rate on Exophase at the case's target, scored by pinball loss on the
logit scale.

Why this beats the house baselines
----------------------------------
The house persistence baseline says every rate stays where it is; the house
curve baseline fits a straight line to logit(rate) against log-time. Both miss
the structure of what an Exophase earn rate IS:

    rate(t) = 100 * E(t) / N(t)

where E(t) is how many tracked players have earned the trophy and N(t) is how
many players Exophase tracks at all. E(t) only ever grows (players keep
finishing chapters and activities), while N(t) grows in bursts as new players
sync their profiles for the first time -- and every new player arrives having
earned almost nothing yet. So the rate can FALL hard even while more and more
players earn the trophy: on launch night the tracked count jumped 638 -> 3,868
at Exophase's first refresh and every rate but the 100% one fell.

This program therefore forecasts the two counts separately and divides:

* N(t): log N against log-time, least squares over the distinct reads.
* E_i(t) per trophy: log E against log-time, least squares over the distinct
  reads (E_i = rate_i * N / 100 from each read's own counts).
* rate* = 100 * E* / N*, with E* floored at its last value (counts never fall)
  and N* floored at its last value, and the fitted log-time slopes damped
  (x0.85) against overshooting over a 4.5-day extrapolation.

Exophase refreshes in batches hours apart, so the history is a step function:
reads are deduplicated to distinct snapshots before fitting, so a long stale
plateau cannot overweight the fit. Only the last 24h of distinct reads are
used when there are at least 3 of them, so the fit follows the current growth
regime, not launch night.

When the history is flat (the practice cases: every read identical, one
distinct snapshot), there is no trend to fit. The launch-night refresh then
supplies the prior -- and it is not the mean rate ratio. The refresh took
players 638 -> 3,868 and trophies earned 943 -> 4,276, a mean ratio of 0.75x,
but nearly all of that earned growth is the first-mission trophy alone
(638 -> 3,868 earners); the other 54 trophies earned 305 -> 408 between them,
so a typical trophy's rate was cut to (408/3868)/(305/638) = 0.22x. The
median is therefore the last rate times ~0.22 (tilted slightly up for the
highest-rate, earliest-content trophies, whose earners grow fastest), with a
wide honest spread. A trophy already at 100% stays ~100%: everyone tracked
has the first-mission trophy.

Spread (logit sd) combines the baseline's random-walk term (0.5 * sqrt(days)),
the fits' residual prediction error, and 0.45 for refresh timing: the answer
is the page as of Exophase's last refresh before the target, which can lag
the target by hours. Floor 0.6, cap 1.7 on the logit scale.

Standard library only, no network, deterministic (no randomness anywhere),
one pass over the case files in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Random-walk spread of the house baseline, logit units per sqrt(day).
RW_SIGMA = 0.5
# Extra logit sd for Exophase's batch-refresh timing / answer-read staleness.
REFRESH_SD = 0.45
# Shrinkage applied to fitted log-time slopes before extrapolating.
SLOPE_DAMP = 0.85
# Flat-history prior: a typical trophy's rate ratio across the launch-night
# refresh (0.2206; see the docstring -- the 0.75 mean is trophy #2 alone).
FLAT_FACTOR = 0.22
# Flat-history prior: how much the factor tilts with the trophy's rate level.
FLAT_TILT = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """x rounded to 4 decimals, half up (as the kit starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, x_mean, sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, mx, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return a, b, resid, mx, sxx


def quantiles(median_pct, sd):
    """Five quantiles around a median percent, on the logit scale."""
    lg = logit(median_pct)
    out = []
    prev = None
    for z in Z:
        q = round4(clamp_pct(inv_logit(lg + z * sd)))
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0

    if not history:
        # No data at all: a flat, very wide forecast. Must never crash.
        return {"trophies": [{"id": t["id"], "q": quantiles(1.0, 1.5)} for t in trophies]}

    # --- distinct snapshots (Exophase refreshes in steps; drop repeats) ---
    snaps = []
    seen = set()
    for h in history:
        rates = tuple(h["rates"])
        key = (h["players_tracked"], rates)
        if key in seen:
            continue
        seen.add(key)
        snaps.append(h)
    last = snaps[-1]
    n_last = last["players_tracked"]
    t0 = parse_time(snaps[0]["at"])
    x_target = math.log1p(max(0.0, (parse_time(case["target"]) - t0) / 3600.0))

    # Last non-null rate per trophy (a read may lack a trophy).
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for h in reversed(snaps):
            if h["rates"][i] is not None:
                r = h["rates"][i]
                break
        last_rate.append(r if r is not None else 0.5)

    if len(snaps) < 2:
        return flat_forecast(trophies, last_rate, horizon_days)

    # --- trend path: fit log N and log E_i against log-time ---
    # Last 24h of distinct reads when there are enough of them, else all.
    cutoff = parse_time(snaps[-1]["at"]) - 24 * 3600
    recent = [h for h in snaps if parse_time(h["at"]) >= cutoff]
    window = recent if len(recent) >= 3 else snaps
    xs = [math.log1p(max(0.0, (parse_time(h["at"]) - t0) / 3600.0)) for h in window]
    n_fit = ols(xs, [math.log(max(1, h["players_tracked"])) for h in window])
    a_n, b_n = n_fit[0], n_fit[1]
    x_mean, sxx = n_fit[3], n_fit[4]
    # Damp the slope by pivoting around the window's centroid (x_mean, y_mean):
    # scaling the slope with the intercept fixed would pivot around x = 0 and
    # collapse the forecast. y_mean = a + b * x_mean.
    y_mean_n = a_n + b_n * x_mean
    log_n_star = y_mean_n + b_n * SLOPE_DAMP * (x_target - x_mean)
    n_star = max(float(n_last), math.exp(log_n_star))
    # Prediction-interval leverage for the N fit.
    lev = 1.0 + (1.0 / len(xs) + ((x_target - x_mean) ** 2 / sxx if sxx > 0 else 1.0))
    sd_n = min(0.8, n_fit[2] * math.sqrt(max(lev, 1.0)))

    rw = RW_SIGMA * math.sqrt(horizon_days)
    rows = []
    for i, t in enumerate(trophies):
        # Earned counts per snapshot from that snapshot's own N. A printed
        # 0.00% means "fewer than half a hundredth of a percent", not zero:
        # use the 0.005% continuity correction, or a 0.00% -> 0.01% rounding
        # step fabricates an explosive log-slope for the rarest trophies.
        e_series = []
        x_series = []
        for h, x in zip(window, xs):
            r = h["rates"][i]
            if r is None:
                continue
            r_adj = r if r > 0 else 0.005
            e_series.append(math.log(max(0.5, r_adj * h["players_tracked"] / 100.0)))
            x_series.append(x)
        r_last_adj = last_rate[i] if last_rate[i] > 0 else 0.005
        e_last = max(0.5, r_last_adj * n_last / 100.0)
        if len(e_series) >= 2 and len(set(x_series)) >= 2:
            a_e, b_e, s_e, _, _ = ols(x_series, e_series)
            # Guardrail: the earned-count slope cannot outrun the tracked
            # count's slope by more than 3 log units (a rounding artefact
            # otherwise lets a 0.00% -> 0.01% step imply absurd growth).
            b_e = min(b_e, b_n + 3.0)
            y_mean_e = a_e + b_e * (sum(x_series) / len(x_series))
            e_star = max(e_last, math.exp(y_mean_e + b_e * SLOPE_DAMP * (x_target - sum(x_series) / len(x_series))))
            sd_e = min(0.9, s_e * math.sqrt(max(lev, 1.0)))
        else:
            # No usable earned-count trend: hold the count, widen the spread.
            e_star = e_last
            sd_e = 0.6
        median = clamp_pct(100.0 * e_star / n_star)
        if last_rate[i] >= 99.5:
            # The first-mission trophy: essentially everyone has it, and
            # the earned count tracks N itself. Keep it at the top.
            median = max(median, 99.0)
        sd = math.sqrt(rw * rw + sd_e * sd_e + sd_n * sd_n + REFRESH_SD * REFRESH_SD)
        sd = min(1.7, max(0.6, sd))
        rows.append({"id": t["id"], "q": quantiles(median, sd)})
    return {"trophies": rows}


def flat_forecast(trophies, last_rate, horizon_days):
    """No trend in the history (practice cases): the refresh prior."""
    # Longer horizons let earners catch up a little with the tracked flood;
    # cap the recovery well below the 0.75 mean ratio (trophy #2 alone).
    factor = FLAT_FACTOR * (1.0 + 0.08 * min(horizon_days, 4.0))
    sd = min(1.6, 0.85 + 0.10 * horizon_days)
    # Typical rate level, for the tilt: earliest content earns fastest.
    present = sorted(r for r in last_rate if 0 < r < 100)
    med_rate = present[len(present) // 2] if present else 1.0
    rows = []
    for t, r in zip(trophies, last_rate):
        if r >= 99.5:
            rows.append({"id": t["id"], "q": quantiles(99.95, 0.3)})
            continue
        f = factor * (max(r, 0.05) / max(med_rate, 0.05)) ** FLAT_TILT
        f = min(1.0, max(0.3, f))
        rows.append({"id": t["id"], "q": quantiles(clamp_pct(r * f), sd)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
