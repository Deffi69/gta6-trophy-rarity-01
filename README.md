# gta6-trophy-rarity-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Deffi69/gta6-trophy-rarity-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Deffi69/gta6-trophy-rarity-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A quantile forecaster for Grand Theft Auto VI's PlayStation 5 trophy earn rates
on Exophase, one week after launch. It reads a case (`/input/<id>.in`: every
recorded read of the game's trophy page before the checkpoint) and writes five
quantiles (5/25/50/75/95%) of each of the 55 trophies' earn rate at the
target, scored by pinball loss on the logit scale.

## The idea

An Exophase earn rate is a ratio, not a count:

```
rate(t) = 100 * E(t) / N(t)
```

`E(t)` — tracked players who earned the trophy — only ever grows, while `N(t)`
— players tracked at all — grows in bursts as new players sync their profiles
for the first time, each arriving having earned almost nothing yet. So the rate
can fall hard even while more players earn the trophy: on launch night the
tracked count jumped 638 → 3,868 at Exophase's first refresh and every rate
but the 100% one fell. The program therefore forecasts the two counts
separately and divides: log `E` and log `N` against log-time, least squares
over the distinct reads (Exophase refreshes in batches, so the history is a
step function — repeats are deduplicated), last-24h window when there are
enough distinct reads, slopes damped around the window centroid against
overshooting a 4.5-day extrapolation. Spread combines a random-walk term, the
fits' residual prediction error, and 0.45 logit sd for refresh timing.

When the history is flat (the practice cases: every read identical), there is
no trend to fit, so the launch-night refresh supplies the prior: a typical
trophy's rate was cut to ~0.22x (the 0.75 mean ratio is the first-mission
trophy alone — it holds 638 of the 943 trophies earned at launch), with a wide
honest spread. Full research notes (live Exophase read, RDR2 analogues) are
in `SOURCES.md` beside this repo in the build workdir.

Deterministic, Python 3.13 standard library only, no network. Runs in ~0.15 s
a case, far inside the 8 s cap.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local test results

Practice cases (against a synthetic answer built from the rules' own refresh
figures — 638 → 3,868 players, 943 → 4,276 trophies earned, every rate but
the 100% one lower; the kit ships no practice answers):

| case | this entry | house baseline (starter) |
|------|-----------|--------------------------|
| p00  | loss 0.254, miss 0.07 pp, 90% coverage 0.98 | loss 1.402, miss 0.68 pp, coverage 0.00 |
| p01  | loss 0.254, miss 0.07 pp, 90% coverage 0.98 | loss 1.402, miss 0.68 pp, coverage 0.00 |

Simulated final (a 2-day history with 4-hour refresh steps, N and earners
growing as power laws to plausible day-7 rates; Ch. 1 22%, final mission
1.5%, platinum 0.08%): loss **0.288** vs starter **0.843**, miss 0.50 pp vs
1.60 pp, coverage 1.00 vs 0.89. Two runs are byte-identical; all edge cases
(no history, one read, 0.00%/100% rates, null rates) produce valid output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=eaffd4np&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
