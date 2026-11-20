# GTA VI trophy rarity forecast

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How to run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network. Reads every `<id>.in` in the
input directory and writes `<id>.out` for each. Deterministic: the same
input writes the same bytes. Runs in ~0.2 s per case.

## Approach

Two regimes, chosen from what the case's history actually shows:

- **Flat / stale history** (fewer than two distinct reads): no trend can be
  estimated, and the page has sat unchanged for hours while the game just
  launched. A refresh is likely before the target, and a refresh dilutes
  every non-universal rate: the tracked population jumps as fresh players
  sync, and fresh players have earned ~1 trophy each. So the forecast is a
  mixture of no-refresh (rates stay) vs refresh (rates diluted toward
  earners / (tracked x dilution)), with the refresh probability from a
  launch-time hazard and the dilution factor a lognormal prior. The
  mixture's quantiles are computed numerically, so the spread is wide
  downwards.
- **Rich history** (two or more distinct reads): players' times to earn a
  trophy are roughly lognormally spread, so each trophy's rate follows an
  adoption curve `r(t) = R * Phi((ln t - m) / 0.8)`, fitted to the distinct
  reads (consecutive identical reads, a stale page between refreshes, are
  collapsed first) and read at the target. The curve saturates at `R`. The
  rise above the last observed rate is shrunk toward zero, harder for
  trophies with few earners; the forecast rate cannot fall below
  last-earners / forecast tracked players (earners never decrease); and the
  platinum is capped at the lowest forecast rate of any other trophy
  (platinum needs every other trophy). The spread combines the fit
  residual, the part of each refresh step the smooth curve did not predict,
  and a drift term growing with the horizon.

A trophy every tracked player holds (100% in every read) is forecast at
~100% with only a small lower tail.

## Local evaluation

No local grade exists for this competition (the scorer and the practice
answers are server-only), so the practice was scored against answers
reconstructed from the refresh facts the kit itself publishes (see
`../work/score.py`): the 04:30Z read had 3,868 players tracked and 4,276
trophies earned, every rate but the 100% one lower, and the old earners
recover exactly from the launch-night rates. The starter baseline scores
~1.38-1.69 on those reconstructions (published: ~1.51); this entry scores
~0.42-0.59. On a synthetic long-horizon case (2 days of reads with
refresh steps, target 4.5 days out), this entry scores 0.14 against the
baseline's 0.36.
