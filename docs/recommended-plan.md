# The Recommended Plan — how it works, and is it legitimate?

A plain-language account of how the plugin turns a stochastic urban-growth simulation
into a single prescriptive "recommended plan", what the numbers say, and an honest
read on whether the approach is sound. All figures are from the Cambourne demo,
reproducible via `uv run python scripts/benchmark_plans.py`.

## TL;DR

- The tool produces **two different things**, and they answer different questions:
  1. **Likelihood layers** — P(built)/P(green)/P(centre) across many simulated futures. This is an *uncertainty* map ("where will development almost certainly go vs where is it contingent").
  2. **The recommended plan** — one concrete, prescriptive layout.
- The recommended plan is built in three steps: **ensemble → consensus → population-aware optimiser**.
- The optimiser is what makes the plan good. It carves an equitable green network into the built fabric, and — as of this change — **pays for that green by densifying the remaining fabric, not by deleting homes** (Isobenefit's "constant inhabitants" principle).
- **Direction: legitimate, with caveats.** It matches what the Isobenefit literature and stochastic-urban-model practice actually call for. The honest caveats are listed at the end; none are blocking.

## What you're looking at on the map

| Layer | What it is | What it answers |
|---|---|---|
| Built / green / centre **likelihood** | Fraction of N simulated runs where each cell ended built / green / centre | *Uncertainty.* High P = robust ("happens under almost any future"); P≈0.5 = contingent |
| **Recommended plan** | One categorical layout: green network / built / centres | *Prescription.* "Here is a coherent, walkable arrangement" |

The key idea: **a probability surface is not a plan.** You can't read a buildable layout off the average of many runs (more on this below) — so the plan is produced separately.

## The pipeline, in three steps

1. **Ensemble.** Run the cellular automaton N times with different random seeds. Each run is a complete, valid city (the growth rules keep green accessible and place centres as it grows). Different seeds put the green/centres in *different but equally valid* places.

2. **Consensus.** Average the runs into probability surfaces and threshold them into a first-cut plan (built where P(built) is high, green where P(green) is high and forms a park ≥ the minimum span). This is a *robust* starting point — it only commits where the runs broadly agree.

3. **Population-aware optimiser.** The consensus alone is mediocre on access (see numbers). So we greedily carve compact parks into the built fabric, always at the spot serving the most currently-unserved homes, until coverage stops improving meaningfully. **Crucially, the green budget is set by densification headroom**: if the fabric houses its people at a mean density and can be densified up to the maximum tier, we may free `1 − mean/max` of the built cells to green and re-house everyone by building the rest denser. Then **centres are placed**: existing centres are kept fixed, and new ones are positioned by facility location — each at the *centroid* of the homes it serves, so it sits central to its catchment rather than on an edge — until everyone is within a walk of a centre.

## The numbers (Cambourne, 800 m walk, all plans scored identically)

| method | served* | green cov | centre cov | worst-off† | built | green |
|---|---|---|---|---|---|---|
| consensus | 36.6% | 45.3% | 82.4% | 0.0% | 7030 | 614 |
| **optimised** | **87.3%** | 97.2% | 89.7% | **15.7%** | 6261 | 1339 |
| single run (median of 16) | 40.8% | 46.4% | 88.9% | 0.0% | 6910 | 718 |

\* *served* = share of homes within an 800 m walk of **both** qualifying green and a centre.
† *worst-off* = benefit of the 5th-percentile (least-served) home; the equity headline.

**Population check (the "constant inhabitants" guarantee):**

```
population held         26,714,000
built cells   7030 -> 6261      (11% freed to green)
mean density  3,800 -> 4,267 / cell   (max 6,000)
feasible by densifying the rest: True
```

So the optimiser added a green network reaching ~97% of homes, brought ~87% within a walk of both green and a centre, and lifted the worst-off from *nothing* to 15.7%, **without losing a single home** — the freed land is compensated by a modest (~12%) density increase that stays comfortably under the maximum tier.

## A correction worth recording

Earlier in this work I claimed a *single* simulation run was far better than the consensus (94% vs 34% served) and nearly recommended shipping one representative run. **That was wrong — an unfair comparison.** The single run had been scored keeping ~100 centres and counting every stray green cell as a park, while the consensus is capped at 50 centres and only counts parks ≥ the minimum span. Scored on equal terms (the table above), a single run is **no better than the consensus** (~37–41%, overlapping ranges). The thing that actually helps is the optimiser. This is exactly why the benchmark now routes *every* candidate through identical scoring.

## Is this a legitimate direction?

**Yes, with the caveats below.** Three reasons it's sound:

1. **It matches the model's intent.** Isobenefit Urbanism (D'Acci) is explicitly designed to produce *many* valid forms, not one canonical plan. Treating the ensemble as uncertainty and presenting a single coherent layout as "one scenario among many valid ones" is faithful; presenting the pixel-average as the plan would not be.
2. **It respects the core principle.** The whole point of Isobenefit is *equal* access to nature and centres while holding inhabitants constant. We now (a) optimise the worst-off, not just the average, and (b) fund green via density, not by deleting housing. Both are central to the paradigm.
3. **It matches stochastic-urban-model practice.** Established CA models (e.g. SLEUTH) report Monte-Carlo probability maps as the primary product and treat any single run as one draw — which is exactly our likelihood-layers + representative-plan split.

## Honest limitations (none blocking)

- **Density is accounted for, not yet drawn.** The plan shows the green network and built footprint; the required density increase is reported (and proven feasible) but not yet painted as per-cell tiers on the map. A future version could colour built by density.
- **Access is a walking proxy.** Distances use an open-grid walk (it doesn't yet route around rivers/barriers), benefit fades linearly to zero at the max distance, and any qualifying park "serves" its catchment regardless of size/quality. More faithful would be distance-decay + network distance + a green size/quality gate. These refine the *score*, not the *approach*.
- **Equity vs land is a policy dial.** How hard to chase the last under-served fringe homes (better worst-off) vs how much to densify (more housing-friendly) is a genuine trade-off, currently set to a balanced default. It could be exposed as a slider.
- **One demo.** All numbers are Cambourne. The method should be checked on a second case before strong general claims.

## Bottom line

The architecture — **ensemble for uncertainty, consensus + population-aware optimiser for the plan** — is coherent, faithful to the Isobenefit principle, and now defensible with a reproducible benchmark. The plan reaches ~97% of homes with green and holds housing constant. The open items are refinements to *fidelity* (better access metric, draw the density), not to the *direction*.
