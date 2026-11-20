// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// A forecaster for Exophase GTA VI trophy earn rates. Two regimes, chosen
// from what the case's history actually shows:
//
//   (A) Flat / stale history (fewer than two distinct reads): no trend can be
//       estimated, and the page has sat unchanged for hours while the game
//       just launched. A refresh is likely before the target, and a refresh
//       dilutes every non-universal rate: the tracked population jumps as a
//       flood of fresh players syncs, and fresh players have earned ~1 trophy
//       each. So the predictive distribution is a mixture: no-refresh (rates
//       stay, random-walk spread) vs refresh (rates diluted toward
//       earners / (tracked * dilution)), with the refresh probability from a
//       launch-time hazard and the dilution factor a lognormal prior. The
//       mixture's quantiles are computed numerically, so the spread is wide
//       downwards and the median sits between the branches.
//
//   (B) Rich history (two or more distinct reads): a lognormal-adoption
//       forecast. Per trophy, players' times to earn it are roughly
//       lognormally spread, so the rate follows an adoption curve
//       r(t) = R * Phi((ln t - m) / 0.8), fitted to the distinct reads by
//       grid search over m with R the least-squares level, and read at the
//       target. The curve saturates at R, unlike a straight line in
//       logit-log-time, which extrapolated from a still-climbing trophy runs
//       past 100%. Counts guardrails keep it honest: the rise above the last
//       observed rate is shrunk toward zero, harder for trophies with few
//       earners; the forecast rate cannot fall below last-earners / forecast
//       tracked players (earners never decrease); and the platinum is capped
//       at the lowest forecast rate of any other trophy (platinum needs every
//       other trophy). The spread combines the fit residual, the part of each
//       refresh step the smooth curve did not predict, and a drift term
//       growing with the horizon.
//       A trophy every tracked player holds (100% in every read) is forecast
//       at ~100% with only a small lower tail.
//
// Reads that repeat the previous read exactly (a stale page between
// refreshes) are collapsed to one distinct read before any fitting, so flat
// stretches cannot overweight the fit. Everything is deterministic: the same
// input writes the same bytes.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Release-day start: time is measured in hours from here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

/** Refresh hazard: mean hours between Exophase refreshes during launch. */
const REFRESH_TAU_H = 1.2;
/** Prior median dilution factor of the tracked count at one refresh. */
const DILUTION_MEDIAN = 5.0;
/** Prior log-spread of the dilution factor. */
const DILUTION_LOG_SD = 0.9;
/** Mean trophies a fresh player has earned beyond the universal trophy. */
const FRESH_EXTRA_TROPHIES = 0.1;
/** Random-walk spread, logit units per square-root day (no-refresh branch). */
const WALK_SIGMA = 0.5;
/** Drift-uncertainty spread, logit units per square-root day (trend branch). */
const DRIFT_SIGMA = 0.3;
/** Spread of players' log-times to earn a trophy (adoption curve width). */
const ADOPTION_SD = 0.8;

interface Trophy {
  id: number;
  tier: string;
}
interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}
interface Case {
  levels: number[];
  horizon_minutes: number;
  at: string;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7). */
const normCdf = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const poly = t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  const p = 1 - Math.exp(-(x * x) / 2) / Math.sqrt(2 * Math.PI) * poly;
  return x >= 0 ? p : 1 - p;
};

/** Least-squares line y ~ a + b*x; returns [intercept, slope]. */
const ols = (xs: number[], ys: number[]): [number, number] => {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
};

const std = (xs: number[]): number => {
  if (xs.length < 2) return 0;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
};

/** Hours from release-day start to an ISO timestamp (floored at 0). */
const hoursSinceRelease = (iso: string): number =>
  Math.max(0, (Date.parse(iso) - RELEASE_MS) / 3_600_000);

/** A read is stale when it repeats the previous read exactly. */
const sameRead = (a: Read, b: Read): boolean =>
  a.players_tracked === b.players_tracked &&
  a.rates.length === b.rates.length &&
  a.rates.every((r, i) => r === b.rates[i]);

/**
 * Quantiles of a two-component normal mixture in logit space, by bisection
 * on the mixture CDF. Branch 1 has weight 1 - p, branch 2 weight p.
 */
const mixtureQuantiles = (
  m1: number, s1: number, m2: number, s2: number, p: number,
): number[] => {
  const cdf = (x: number): number =>
    (1 - p) * normCdf((x - m1) / s1) + p * normCdf((x - m2) / s2);
  const lo = logit(0.01);
  const hi = logit(99.99);
  return LEVELS.map((tau) => {
    let a = lo;
    let b = hi;
    for (let k = 0; k < 80; k++) {
      const mid = (a + b) / 2;
      if (cdf(mid) < tau) a = mid;
      else b = mid;
    }
    return round4(invLogit((a + b) / 2));
  });
};

/** Quantiles of a single normal in logit space around a median percent. */
const normalQuantiles = (medianPct: number, sigma: number): number[] => {
  const m = logit(medianPct);
  return Z.map((z) => round4(clampPct(invLogit(m + z * sigma))));
};

/** The latest non-null rate for trophy i, else null. */
const lastRate = (c: Case, i: number): number | null => {
  for (let k = c.history.length - 1; k >= 0; k--) {
    const r = c.history[k].rates[i];
    if (r !== null && r !== undefined) return r;
  }
  return null;
};

const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  const horizonDays = c.horizon_minutes / 1440;
  const horizonH = c.horizon_minutes / 60;
  const last = c.history[c.history.length - 1];
  const nLast = last.players_tracked;

  // Collapse consecutive identical reads: the page between refreshes.
  const distinct: Read[] = [];
  for (const r of c.history) {
    if (distinct.length === 0 || !sameRead(distinct[distinct.length - 1], r)) {
      distinct.push(r);
    }
  }

  // Earners per trophy in the last read (integers; rates print to 2 decimals).
  const earnersLast: Array<number | null> = c.trophies.map((t, i) => {
    const r = lastRate(c, i);
    return r === null ? null : Math.round((r / 100) * nLast);
  });
  // A trophy every tracked player holds: the refresh cannot dilute it.
  const universal: boolean[] = c.trophies.map((t, i) => {
    const r = lastRate(c, i);
    return r !== null && r >= 99.99;
  });
  const universalRow = (): number[] => [99.9, 99.99, 100, 100, 100];

  if (distinct.length < 2) {
    // ---- Regime A: flat / stale history. Refresh-mixture forecast. ----
    const pRefresh = Math.min(0.9, Math.max(0.05, 1 - Math.exp(-horizonH / REFRESH_TAU_H)));
    const walk = WALK_SIGMA * Math.sqrt(horizonDays);
    const sigmaB = Math.sqrt(walk * walk + DILUTION_LOG_SD * DILUTION_LOG_SD);
    // New-cohort earn shares: the universal trophy is earned by every fresh
    // player; the remaining ~0.1 trophies a fresh player holds are spread
    // over the rest, sharpened toward the trophies the old cohort earned
    // most (the earliest ones).
    let sumSq = 0;
    for (let i = 0; i < c.trophies.length; i++) {
      if (!universal[i] && earnersLast[i] !== null) {
        sumSq += earnersLast[i]! * earnersLast[i]!;
      }
    }
    return c.trophies.map((t, i) => {
      if (universal[i]) return { id: t.id, q: universalRow() };
      const r = lastRate(c, i);
      if (r === null) return { id: t.id, q: [0.01, 0.01, 0.02, 0.1, 1] };
      const e = earnersLast[i] ?? 0;
      const share = sumSq > 0 ? (FRESH_EXTRA_TROPHIES * e * e) / sumSq : 0;
      const freshEarners = (DILUTION_MEDIAN - 1) * nLast * share;
      const rRefresh = ((e + freshEarners) / (nLast * DILUTION_MEDIAN)) * 100;
      const q = mixtureQuantiles(logit(r), walk, logit(rRefresh), sigmaB, pRefresh);
      return { id: t.id, q };
    });
  }

  // ---- Regime B: rich history. Lognormal-adoption forecast.
  //
  // A trophy's earn rate is the share of tracked players who have earned it,
  // and players' times to earn a trophy are roughly lognormally spread, so
  // the rate follows an adoption curve r(t) = R * Phi((ln t - m) / s0):
  // R the eventual share who will ever earn it, m the median log-time to
  // earn it, s0 the spread of players' paces (fixed at 0.8). Per trophy the
  // curve is fitted to the distinct reads by grid search over m, with R the
  // least-squares level for that m, and read at the target. The curve
  // saturates at R, which is what a straight line in logit-log-time cannot
  // do: a line extrapolated from a still-climbing trophy runs past 100%,
  // while the adoption curve bends over as the trophy's audience is
  // exhausted. Headroom above the last observed rate is shrunk toward zero,
  // harder for trophies with few earners, whose curves are noise; the
  // forecast rate is floored at last-earners / forecast tracked (earners
  // never decrease) and the platinum is capped at the lowest forecast rate
  // of any other trophy (platinum needs every other trophy).
  const xTarget = Math.log(1 + hoursSinceRelease(c.target));
  const uTarget = Math.log(Math.max(0.5, hoursSinceRelease(c.target)));
  const xs = distinct.map((r) => Math.log(1 + hoursSinceRelease(r.at)));
  const us = distinct.map((r) => Math.log(Math.max(0.5, hoursSinceRelease(r.at))));
  // Tracked-players power law over all distinct reads (for the earners floor).
  const logN = distinct.map((r) => Math.log(Math.max(1, r.players_tracked)));
  const [aN, bNraw] = ols(xs, logN);
  const bN = Math.min(2, Math.max(0, bNraw));
  const nTarget = Math.max(nLast, Math.exp(aN + bN * xTarget));
  const horizonDaysB = horizonDays;

  const medians: number[] = [];
  const sigmas: number[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    if (universal[i]) {
      medians.push(100);
      sigmas.push(0.1);
      continue;
    }
    const eLast = earnersLast[i] ?? 0;
    const pts: Array<{ u: number; r: number }> = [];
    for (let k = 0; k < distinct.length; k++) {
      const rate = distinct[k].rates[i];
      if (rate === null || rate === undefined) continue;
      pts.push({ u: us[k], r: rate });
    }
    if (pts.length < 2) {
      // Too little signal: hold the last rate, wide spread.
      medians.push(clampPct(lastRate(c, i) ?? 0.01));
      sigmas.push(1.2);
      continue;
    }
    // Grid search over the median log-time m; R is least squares for that m.
    const rMax = Math.max(...pts.map((p) => p.r));
    let best: { sse: number; R: number; m: number } | null = null;
    const mGrid = 140;
    for (let g = 0; g < mGrid; g++) {
      const m = Math.log(0.5) + ((Math.log(500) - Math.log(0.5)) * g) / (mGrid - 1);
      let num = 0;
      let den = 0;
      const phis = pts.map((p) => normCdf((p.u - m) / ADOPTION_SD));
      for (let k = 0; k < pts.length; k++) {
        num += pts[k].r * phis[k];
        den += phis[k] * phis[k];
      }
      let R = den > 0 ? num / den : rMax;
      R = Math.min(100, Math.max(rMax, R));
      let sse = 0;
      for (let k = 0; k < pts.length; k++) {
        const d = pts[k].r - R * phis[k];
        sse += d * d;
      }
      if (best === null || sse < best.sse) best = { sse, R, m };
    }
    const fit = best!;
    // Shrink the rise above the last observed rate toward zero.
    const w = eLast / (eLast + 20);
    const rLast = lastRate(c, i) ?? 0.01;
    const rTraw = fit.R * normCdf((uTarget - fit.m) / ADOPTION_SD);
    let rT = clampPct(rLast + w * Math.max(0, rTraw - rLast));
    // Earners floor: the rate cannot fall below last-earners / forecast tracked.
    const floorPct = (eLast / nTarget) * 100;
    if (rT < floorPct) rT = clampPct(floorPct);
    medians.push(rT);
    // Spread: fit residual + unexpected refresh-step volatility + horizon drift.
    const resid = pts.map((p) => {
      const f = clampPct(fit.R * normCdf((p.u - fit.m) / ADOPTION_SD));
      return logit(p.r) - logit(f);
    });
    const sigmaResid = Math.max(0.15, std(resid));
    // Only the part of each step the smooth curve did not predict counts as
    // refresh volatility: the raw step size is mostly the trend itself.
    const jumpResid: number[] = [];
    for (let k = 1; k < pts.length; k++) {
      const obs = logit(pts[k].r) - logit(pts[k - 1].r);
      const fitStep =
        logit(clampPct(fit.R * normCdf((pts[k].u - fit.m) / ADOPTION_SD))) -
        logit(clampPct(fit.R * normCdf((pts[k - 1].u - fit.m) / ADOPTION_SD)));
      jumpResid.push(obs - fitStep);
    }
    const sigmaJump = jumpResid.length > 0 ? std(jumpResid) / Math.SQRT2 : 0.25;
    const sigmaDrift = DRIFT_SIGMA * Math.sqrt(horizonDaysB);
    sigmas.push(
      Math.min(2.5, Math.max(0.35, Math.sqrt(sigmaResid * sigmaResid + sigmaJump * sigmaJump + sigmaDrift * sigmaDrift))),
    );
  }

  // Platinum needs every other trophy: cap its forecast rate at the lowest
  // forecast rate of any other trophy.
  const platIdx = c.trophies.findIndex((t) => t.tier === 'platinum');
  if (platIdx >= 0 && !universal[platIdx]) {
    let minOther = Infinity;
    for (let i = 0; i < c.trophies.length; i++) {
      if (i !== platIdx && !universal[i]) minOther = Math.min(minOther, medians[i]);
    }
    if (Number.isFinite(minOther) && medians[platIdx] > minOther) {
      medians[platIdx] = minOther;
    }
  }

  return c.trophies.map((t, i) => {
    if (universal[i]) return { id: t.id, q: universalRow() };
    return { id: t.id, q: normalQuantiles(medians[i], sigmas[i]) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
