# Experiment A -- noise kinds x rates (nested-CV, linear, 10 datasets)

Question: does bounded loss (wave) pull ahead of hard GBTSVM at high / structured
noise, where a redescending loss should help most?

## Answer: the wave-hard gap does NOT grow with noise -- it is flat and small.

wave-hard mean gap by rate: 0.3 -> +0.010, 0.4 -> +0.009, 0.5 -> +0.008.
So bounded loss gives a small, consistent edge (~+1pp) that does not scale.
It is slightly larger on outlier-type noise (far +0.012, boundary +0.012) than on
symmetric (+0.005) -- directionally consistent with theory, but tiny.

win-count wave>hard: symmetric 14/30, asymmetric 19/30, boundary 20/30, far 18/30.

## The real finding is about binom, not wave.

binom-hard gap by kind x rate:
- symmetric: POSITIVE and grows (+0.027 / +0.020 / +0.052) -- binom's home turf.
- asymmetric: strongly NEGATIVE and worsens (-0.046 / -0.134 / -0.300); binom
  degenerates (one-sided flips break the binomial iid assumption -> wrong stop).
- boundary/far: roughly neutral.

So: binom is fast + best on SYMMETRIC noise, but CATASTROPHICALLY fragile under
ASYMMETRIC noise. wave never collapses -- it stays ~+1pp above hard across all
four kinds and all rates.

## Reframed opportunity

wave's value is CONSISTENCY / robustness, not a large average gain. The natural
paper claim: bounded loss makes the fast granular-ball approach SAFE under the
asymmetric / structured noise that breaks adaptive stopping alone. Decisive next
test (B): does wave2 (binom balls + wave loss) avoid binom's asymmetric collapse
while keeping its speed?

File: 260913_022641_noisekinds_linear.csv
