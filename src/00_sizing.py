"""
Stage 0 - measurement resolution, computed before the task is chosen.

Project 2 had one noise source: decoding. Temperature 0 plus a pinned provider
nearly killed it, and what was left (2.7 pts) was measured directly.

Project 4 has two, and the second one does not go away with temperature 0:

    decoding noise   - same weights, same prompt, different sample
    training noise   - same data, same hyperparameters, different seed

Training noise is bounded below by nothing we control. It is measured by
training the same config k times, which costs k times as much GPU. So the
question "how many test items?" cannot be answered without also answering
"how many seeds?", and past a certain point more test items buy nothing.

This script answers three things:

  A. Unpaired accuracy CI - the width of the error bar on a single number.
  B. Paired minimum detectable difference (McNemar) - the smallest delta
     between two configs we could call real at 80% power.
  C. The seed-noise ceiling - the point where more test items stop helping
     and only more seeds do.

No model, no dataset, no GPU. Requires numpy.
"""

import math
import numpy as np

Z_ALPHA = 1.959963985  # two-sided 95%
Z_BETA = 0.8416212336  # 80% power

TEST_SIZES = [75, 150, 300, 500, 800, 1200]
RNG = np.random.default_rng(0)


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------- A
def unpaired_ci():
    rule("A. 95% CI half-width on a single accuracy number (percentage points)")
    accs = [0.50, 0.70, 0.85]
    print("      n  " + "  ".join(f"p={a:.2f}" for a in accs))
    for n in TEST_SIZES:
        cells = []
        for p in accs:
            half = Z_ALPHA * math.sqrt(p * (1 - p) / n) * 100
            cells.append(f"{half:6.1f}")
        print(f"{n:7d}  " + "  ".join(cells))
    print("\n  This is the error bar on 'LoRA scored 84.2%'. One decimal place")
    print("  is unreportable at every n in this table.")


# ---------------------------------------------------------------- B
def paired_mdd():
    rule("B. Minimum detectable difference, paired (McNemar), 80% power (pp)")
    print("  Discordance = share of test items the two configs disagree on.")
    print("  Small interventions (rank, LR) disagree on few items; a glossary-")
    print("  sized intervention disagrees on many.\n")
    psis = [0.05, 0.10, 0.20, 0.35]
    print("      n  " + "  ".join(f"psi={p:.2f}" for p in psis))
    for n in TEST_SIZES:
        cells = []
        for psi in psis:
            mdd = (Z_ALPHA + Z_BETA) * math.sqrt(psi / n) * 100
            cells.append(f"{8 * ' '}"[: 8 - len(f"{mdd:.1f}")] + f"{mdd:.1f}")
        print(f"{n:7d}  " + "  ".join(cells))
    print("\n  Read the n=75 row before deciding to reuse the Olist question set.")


# ---------------------------------------------------------------- C
def seed_ceiling(reps=4000, p_base=0.80, true_delta=0.03, sigma_item=1.0):
    rule("C. Where seed noise caps resolution")
    print(f"  Simulated: base accuracy {p_base:.0%}, true effect "
          f"{true_delta*100:.0f} pp, shared test items (paired).")
    print("  sigma_seed = SD of a config's true accuracy across training seeds.\n")

    logit = lambda x: math.log(x / (1 - x))
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    scale = p_base * (1 - p_base)  # delta-method: accuracy pts -> logits

    a_base = logit(p_base)
    a_treat = logit(p_base + true_delta)

    print("  sigma_seed  seeds  " + "  ".join(f"n={n}" for n in [150, 500, 1200]))
    for sigma_acc in [0.000, 0.010, 0.020, 0.030]:
        for k in [1, 3]:
            cells = []
            for n in [150, 500, 1200]:
                d_item = RNG.normal(0, sigma_item, size=n)  # fixed test set
                s = sigma_acc / scale
                diffs = np.empty(reps)
                for r in range(reps):
                    acc = []
                    for a in (a_base, a_treat):
                        offs = RNG.normal(a, s, size=k)[:, None]
                        pc = sigmoid(offs - d_item[None, :])
                        acc.append((RNG.random((k, n)) < pc).mean())
                    diffs[r] = acc[1] - acc[0]
                se = diffs.std(ddof=1) * 100
                power = (np.abs(diffs) > Z_ALPHA * diffs.std(ddof=1)).mean()
                cells.append(f"{se:4.1f}pp/{power*100:3.0f}%")
            print(f"  {sigma_acc*100:8.1f}pp  {k:5d}  " + "  ".join(cells))
    print("\n  Each cell: SE of the measured delta / power to call it real.")
    print("  Follow a row across: past the point where sigma_seed dominates,")
    print("  tripling the test set moves nothing. Only seeds move it.")


if __name__ == "__main__":
    unpaired_ci()
    paired_mdd()
    seed_ceiling()
    print("\nAssumptions worth arguing with: item-difficulty SD of 1.0 logit,")
    print("normal seed effects, seeds independent. All three are guesses until")
    print("we measure sigma_seed for real, which is stage 3.\n")
