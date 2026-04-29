# Learnings From the T-BIOM Reviews

## Overall learnings

The reviews show that the paper's main issue is not only whether the method works. The bigger issue is that the manuscript makes the reader do too much interpretive work. Important details are present in parts, but they are not made explicit enough, and some claims are stronger than what the tables support.

There are five broad lessons:

1. A strong method is not enough if the protocol is not completely reproducible.
2. A hard evaluation setting must be defined very carefully, otherwise reviewers will reinterpret it in a weaker or different way.
3. Novelty claims need direct evidence, not just intuition.
4. Conclusions must match the tables exactly; one overstated sentence can damage trust in the whole paper.
5. Stability evidence such as multiple seeds and 95% confidence intervals is valuable, but reviewers may still distinguish that from a formal pairwise significance test.

## Reviewer 1 learnings

Reviewer 1 is teaching that reproducibility and protocol transparency are part of technical soundness, not secondary details.

### What Reviewer 1 reacted to

- broken code link
- missing number of random seeds
- unclear bad-quality sample removal in FV-300
- unclear training and validation split
- unclear ablation setup
- weak baseline documentation
- figure and wording issues

### Learning from Reviewer 1

The paper currently assumes that if the main idea and results are visible, readers will accept the rest. Reviewer 1 is saying the opposite: if the experiment cannot be reconstructed exactly, then the results become questionable even if the method itself seems reasonable.

The deeper lesson is that every statement about data handling and evaluation must be auditable:

- if samples were removed, say exactly how many and why
- if seeds were used, say how many
- if validation splits vary, define the rule
- if an ablation uses one scenario, say which one
- if code will be released, the link must work

Reviewer 1 is not fundamentally rejecting the idea. The review says the paper is generally well written and technically detailed. The criticism is that missing reproducibility details make the work hard to trust.

### What this means for the revision

The revision must treat reproducibility as a first-class contribution:

- exact protocol
- exact split rules
- exact data curation notes
- exact baseline settings
- working repository

## Reviewer 2 learnings

Reviewer 2 is teaching that framing and terminology matter as much as results, especially when the task is nonstandard.

### What Reviewer 2 reacted to

- confusion between open-set verification and cross-dataset evaluation
- concern about fairness of baseline comparison
- insufficient justification of DSConv adaptation
- insufficient justification of the proposed loss
- lack of computational-cost analysis
- limited analysis of generalization and VERA degradation

### Learning from Reviewer 2

The manuscript currently expects the reader to accept that leave-one-dataset-out with disjoint identities is self-evidently open-set. Reviewer 2 shows that this assumption is unsafe. Your protocol is already open-set with respect to training identities, because the held-out dataset contains only unseen subjects. The problem is that the manuscript does not explicitly separate this `subject-disjoint open-set` meaning from the stricter `non-enrolled identity rejection` meaning that some reviewers associate with open-set deployment.

There is also a second lesson: strong results alone do not guarantee convincing evaluation. If handcrafted methods beat several learned methods on some splits, the paper must proactively explain why. Otherwise reviewers will suspect unfair implementation, poor tuning, or inconsistent protocols.

Reviewer 2 is also pushing the paper toward a more complete journal-level standard:

- define the task rigorously
- distinguish subject-disjoint open-set verification from enrollment-based unknown rejection
- compare against stronger loss baselines
- report operational metrics such as GAR or GMR at fixed FAR or FMR
- discuss deployment cost, not only accuracy
- analyze why the model generalizes better

### What this means for the revision

The revision must make the paper harder to misread:

- define open-set precisely
- state clearly that the current protocol is already open-set with respect to training identities
- distinguish cross-dataset subject-disjoint evaluation from enrollment-based unknown rejection and from standard intra-database verification
- document baseline fairness explicitly
- add complexity/runtime discussion
- explain failure modes, especially VERA

## Reviewer 3 learnings

Reviewer 3 is teaching that T-BIOM is holding the paper to a high bar on novelty, scientific insight, and claim discipline.

### What Reviewer 3 reacted to

- architecture looks like a straightforward combination of known parts
- loss novelty is unclear relative to ArcFace/CosFace/SphereFace
- claims of superiority are inconsistent with Table 5
- genuine and impostor pair generation is unclear
- cross-dataset results alone are not enough for a practical verification claim
- ablation is too shallow

### Learning from Reviewer 3

Reviewer 3 is essentially saying: "Do not ask me to infer the scientific contribution. Show it directly."

The paper currently relies too much on the idea that combining DSConv, GNN, and the proposed loss is obviously meaningful. Reviewer 3 does not accept that. From this review, the main lesson is that novelty by combination is weak unless the paper demonstrates:

- why these components belong together
- what each one contributes
- why the loss is better than standard angular-margin losses
- which claims are global and which are only conditional

Reviewer 3 also flags a trust issue. Once the reviewer sees a claim like "outperforms handcrafted and deep learning approaches across five datasets" and then sees Table 5 contradict it, the credibility of the rest of the manuscript drops.

This reviewer is not only asking for more experiments. The reviewer is asking for tighter scientific argument:

- more honest claims
- clearer scope
- stronger ablation logic
- better connection between method design and problem definition

The other useful lesson from Reviewer 3 is that pairwise subject-disjoint evaluation and gallery-based unknown rejection are being mentally conflated. The manuscript has to separate those explicitly so that the reviewer does not read the current protocol as if it were merely closed-set cross-dataset testing.

### What this means for the revision

The revision must stop defending novelty rhetorically and start proving it empirically:

- direct comparison with ArcFace and CosFace
- component ablation on DSConv and graph modeling
- more precise conclusions
- explicit explanation of why FV-300 behaves differently

## Combined learning across reviewers

All three reviewers are pointing to the same underlying lesson:

The manuscript is closest to being accepted when it is more precise, less ambitious in its wording, and more explicit in its evidence.

The main combined learnings are:

1. The paper should claim `cross-dataset subject-disjoint open-set verification` explicitly, and then separate that from `enrollment-based unknown-identity rejection` rather than using one overloaded phrase for both.
2. The paper should claim `strong overall generalization` rather than `dominates all baselines on every dataset`.
3. The paper should present the proposed loss as a hypothesis that is now validated against stronger losses, not as a self-evident contribution.
4. The paper should make reproducibility visible in the manuscript itself, not leave it to the future repository.
5. The paper should present the existing 5-seed 95% confidence intervals as evidence of stability, while avoiding pairwise `statistically significant` claims unless a formal significance test is added.

## Best takeaway for the meeting

If you want one concise summary for Thursday, it is this:

The reviewers are not saying the topic is uninteresting. They are saying the paper currently asks them to trust too much and verify too little, and it does not distinguish clearly enough between two different meanings of open-set verification.

That can be fixed by:

- defining the task better
- explaining why the current evaluation is already subject-disjoint open-set
- optionally adding one stricter enrollment-based unknown-rejection experiment
- reducing overstated claims
- adding the missing standard verification experiments
- expanding the ablation so the contribution becomes visible
- making the whole protocol fully reproducible


| Dataset | Method | AUC (%) (95% CI) | EER (%) (95% CI) | TAR@FAR=0.1% (%) (95% CI) | TAR@FAR=1% (%) (95% CI) | TAR@FAR=10% (%) (95% CI) |
|---|---|---|---|---|---|---|
| fv300 | proposed | 99.99 (99.98--100.00) | 0.39 (0.06--0.72) | 98.66 (96.85--100.00) | 99.77 (99.41--100.00) | 100.00 (100.00--100.00) |
| fv300 | lgfin | 99.99 (99.98--100.00) | 0.43 (0.09--0.78) | 98.42 (96.98--99.86) | 99.77 (99.44--100.00) | 100.00 (100.00--100.00) |
| fv300 | fvit | 96.55 (95.04--98.06) | 9.60 (7.59--11.61) | 29.05 (7.24--50.86) | 48.83 (28.26--69.40) | 90.48 (85.25--95.70) |
| fv300 | arcvein | 96.36 (94.01--98.72) | 11.03 (5.95--16.11) | 54.26 (45.30--63.22) | 67.25 (57.34--77.15) | 89.13 (81.94--96.31) |
| fv300 | veinAttNet | 99.73 (99.63--99.83) | 2.43 (1.81--3.05) | 63.32 (47.63--79.01) | 93.71 (90.81--96.60) | 99.90 (99.86--99.94) |
| fv300 | resnet | 99.98 (99.96--100.00) | 0.46 (0.03--0.90) | 97.29 (91.97--100.00) | 99.69 (99.18--100.00) | 100.00 (100.00--100.00) |

| Dataset | Method | AUC (%) (95% CI) | EER (%) (95% CI) | TAR@FAR=0.1% (%) (95% CI) | TAR@FAR=1% (%) (95% CI) | TAR@FAR=10% (%) (95% CI) |
|---|---|---|---|---|---|---|
| fvusm | proposed | 98.26 (97.48--99.04) | 7.06 (5.40--8.72) | 53.38 (41.30--65.46) | 73.66 (67.37--79.94) | 95.33 (92.58--98.08) |
| fvusm | lgfin | 95.60 (94.71--96.48) | 11.25 (9.89--12.61) | 28.91 (7.57--50.25) | 45.42 (27.85--62.99) | 87.17 (83.96--90.39) |
| fvusm | fvit | 82.90 (79.16--86.65) | 26.25 (23.52--28.98) | 5.72 (0.00--14.46) | 15.93 (3.59--28.26) | 45.14 (33.70--56.58) |
| fvusm | arcvein | 65.05 (61.48--68.62) | 38.01 (35.77--40.24) | 0.69 (0.00--1.64) | 2.69 (1.08--4.30) | 18.54 (15.01--22.07) |
| fvusm | veinAttNet | 83.85 (77.20--90.51) | 25.60 (20.08--31.11) | 10.73 (3.40--18.05) | 19.45 (6.84--32.06) | 50.97 (34.37--67.56) |
| fvusm | resnet | 81.49 (79.56--83.42) | 26.50 (24.39--28.61) | 0.90 (0.40--1.41) | 5.06 (3.25--6.88) | 44.47 (40.64--48.30) |

| Dataset | Method | AUC (%) (95% CI) | EER (%) (95% CI) | TAR@FAR=0.1% (%) (95% CI) | TAR@FAR=1% (%) (95% CI) | TAR@FAR=10% (%) (95% CI) |
|---|---|---|---|---|---|---|
| mmcbnu | proposed | 99.31 (99.13--99.50) | 4.07 (3.43--4.71) | 49.24 (30.59--67.90) | 85.92 (82.15--89.69) | 98.91 (98.45--99.37) |
| mmcbnu | lgfin | 99.19 (98.96--99.42) | 4.33 (3.48--5.17) | 52.17 (41.93--62.41) | 84.43 (80.43--88.44) | 98.50 (97.78--99.22) |
| mmcbnu | fvit | 92.20 (91.14--93.27) | 16.25 (14.99--17.51) | 21.14 (14.92--27.36) | 40.06 (35.14--44.97) | 74.98 (71.56--78.40) |
| mmcbnu | arcvein | 78.51 (72.71--84.30) | 29.70 (24.39--35.00) | 4.79 (0.83--8.75) | 9.20 (3.09--15.30) | 30.04 (20.11--39.96) |
| mmcbnu | veinAttNet | 92.52 (87.99--97.04) | 15.00 (9.46--20.54) | 10.71 (0.21--21.20) | 39.39 (21.50--57.28) | 79.25 (65.82--92.68) |
| mmcbnu | resnet | 98.74 (97.66--99.81) | 5.27 (2.78--7.76) | 43.58 (19.65--67.50) | 76.24 (60.48--91.99) | 97.42 (94.35--100.00) |

