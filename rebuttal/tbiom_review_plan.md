# T-BIOM Review Action Plan for OpenVeinNet

## Bottom line

The paper is still salvageable because the EiC explicitly allowed a major revision despite the AE recommendation. The common problem across all three reviews is not only performance, but clarity and rigor:

- the manuscript over-claims in a few places
- the evaluation protocol is not explicit enough
- the loss-function novelty is under-justified
- the main architectural choices are not ablated directly
- reproducibility details are incomplete

Reviewer 3 is the most negative, but many of Reviewer 3's points overlap with Reviewer 1 and Reviewer 2. That is good news, because a focused revision can address several reviews at once.

## TL;DR: Additional experiments to run

These are the actual extra experiments and measurements still worth adding beyond your current leave-one-dataset-out subject-disjoint evaluation.

1. Loss-function comparison.
   - Compare your proposed loss against:
     - Softmax / CrossEntropy
     - ArcFace
     - CosFace
     - optional SphereFace

2. Beta sensitivity study for the proposed loss.
   - Example values: `0, 0.1, 0.3, 0.5, 1.0`

3. True component ablation.
   - Full model: DSConv stem + Grapher backbone + proposed loss
   - Replace only the DSConv stem with a standard convolutional stem, while keeping the backbone, embedding size, loss, and protocol fixed
   - Replace only the Grapher backbone with a parameter-matched convolutional backbone, while keeping the stem, embedding size, loss, and protocol fixed
   - Optional: disable the graph operation inside the backbone while keeping the surrounding block structure unchanged, if that is easier to implement cleanly

4. Paired `t`-test against the strongest learned baseline.
   - Most likely OpenVeinNet vs VeinAttNet
   - Use the same 5 seeds
   - Report mean difference and `p`-value for EER, and optionally AUC

5. Runtime / complexity measurement.
   - Parameter count
   - Model size
   - GPU latency
   - CPU latency
   - Optional FLOPs

If you need the minimum viable set for revision, it is:
- ArcFace/CosFace loss comparison
- component ablation
- paired `t`-test
- runtime table
- and stronger protocol clarification in the paper/rebuttal

## Easy step-by-step revision plan

This is the simplest execution order based on what is currently written in `paper.tex`.

### Step 1. Fix the paper's story about the evaluation setting

What to change:
- Rewrite the abstract, introduction, Section 3.2, and conclusion so that the current protocol is described as `cross-dataset subject-disjoint open-set verification`.
- Explicitly say that:
  - the model is trained on four datasets and tested on the fifth
  - identities are disjoint across datasets
  - every image in the held-out dataset is used once as an enrolment template
  - it is compared against all other held-out images
  - same-identity pairs are genuine
  - different-identity pairs are impostor
  - self-comparisons are excluded
  - no pair subsampling is performed

Where in `paper.tex`:
- abstract: around lines 479--481
- introduction: around lines 614--620
- protocol section: around lines 1052--1056
- conclusion: around line 1260

What this addresses:
- R2.1 open-set vs cross-dataset confusion
- R3.4 unclear pair construction
- part of the EiC concern about clarity

### Step 2. Remove the overclaims that are directly contradicted by Table 5

What to change:
- Replace all versions of:
  - `consistently outperforms existing techniques`
  - `consistently outperforms both handcrafted and deep learning methods`
  - `state-of-the-art across diverse open-set conditions`
- Replace with claims like:
  - best overall cross-dataset generalization among learned methods
  - strongest performance on the most challenging held-out datasets
  - near-best or best learned performance on 4/5 evaluation scenarios
- Add one explicit sentence explaining why handcrafted methods are stronger on FV-300.

Where in `paper.tex`:
- abstract: line 481
- introduction: line 615
- results: lines 1132--1148
- conclusion: line 1260

What this addresses:
- R3.3 directly
- R2.2 indirectly
- overall credibility problem

### Step 3. Fix the source-level inconsistencies inside the method description

What to change:
- Figure caption currently says `Graph-Augmented Softmax Loss`; align it with `Centroid Angular Hybrid Loss`.
- Introduction currently says the final verification stage uses a `single-layer CNN` with a `softmax classifier`, but Section 3 says verification is done using cosine similarity on embeddings.
- Make the method description consistent everywhere with the actual implemented evaluation.

Where in `paper.tex`:
- Figure 2 caption: line 598
- introduction/method summary: line 615
- protocol section: lines 1052--1053

What this addresses:
- R1 reproducibility concerns
- R3 trust/consistency concerns
- general reviewer confusion

### Step 4. Make Section 3.2 fully reproducible

What to change:
- State the exact number of random seeds: `5`
- State the seed values if available
- State whether the CIs are over seeds only
- Define the training/validation split rule
- Define whether split variation changes across seeds
- State exactly how many FV-300 samples were removed and the criterion for removal
- Add the exact formulas or definitions used to compute genuine and impostor counts

Where in `paper.tex`:
- Table 3 and surrounding dataset text: lines 916--946
- Table 4 and protocol text: lines 1031--1056

What this addresses:
- R1.2
- R1.3
- R1.4
- R3.4

### Step 5. Add the paired t-test properly

What to change:
- Add a short statistics paragraph in the results section or a dedicated small subsection.
- Use a paired `t`-test between OpenVeinNet and the strongest learned baseline, most likely VeinAttNet.
- Pair the comparison by seed under the same evaluation split.
- Do this for each held-out dataset, and optionally also on the mean EER across the five held-out datasets.
- Report:
  - mean difference
  - `p`-value
  - which metric is tested: EER and/or AUC
- Keep the existing 95\% CI discussion and clarify that:
  - CIs show stability across 5 runs
  - paired `t`-tests assess whether the difference between two methods is statistically reliable

Where in `paper.tex`:
- results discussion around lines 1132--1148
- possibly a new paragraph immediately after Table 5

What this addresses:
- R2.5 directly
- supports stronger comparative claims

Practical note:
- with only 5 seeds, the paired `t`-test is acceptable if you are careful, but do not oversell it
- use wording like `the improvement is statistically significant at the 0.05 level` only where the test actually supports it

### Step 6. Strengthen the loss-function evidence

What to change:
- Expand the current loss ablation beyond MSE and NLL
- Add:
  - Softmax / CrossEntropy baseline
  - ArcFace
  - CosFace
  - optional SphereFace
- Add a beta sensitivity study
- Rewrite the explanation of the loss in plain language:
  - what the angular term does
  - why the frozen auxiliary weights help
  - why this is useful under subject-disjoint verification

Where in `paper.tex`:
- loss section: around lines 861--896
- loss ablation table and text: lines 1197--1210

What this addresses:
- R2.3
- R3.2

### Step 7. Turn the refinement study into a true ablation study

What to change:
- Keep the kernel-size and GrapherBlock refinements if you want
- Add component ablations:
  - full model
  - standard convolution stem in place of DSConv, with the rest of the pipeline unchanged
  - parameter-matched convolutional backbone in place of the Grapher backbone, with the rest of the pipeline unchanged
  - optional: disable only the graph operation inside the backbone if that yields a cleaner one-factor ablation
- State clearly which dataset and protocol every ablation uses

How to keep this apples-to-apples:
- change one component at a time
- keep the embedding dimensionality fixed
- keep the training schedule, optimizer, and loss fixed
- keep the evaluation protocol fixed
- if you replace the backbone, try to keep depth/width roughly matched rather than switching to a completely different-size model

Where in `paper.tex`:
- ablation section around lines 1185--1210

What this addresses:
- R1.6
- R3.1
- R3.6

### Step 8. Add a fairness and deployment paragraph

What to change:
- Add one short paragraph explaining why handcrafted methods can be stronger on FV-300
- Add one short paragraph explaining the VERA drop
- Add a small runtime table:
  - params
  - model size
  - GPU latency
  - CPU latency
  - optional FLOPs

Where in `paper.tex`:
- results discussion around lines 1136--1148
- add a small table in experiments or discussion

What this addresses:
- R2.2
- R2.4
- R2.5
- R3.3

### Step 9. Fix all low-level presentation issues before resubmission

What to change:
- Fix grammar:
  - `The architecture in paper describes a an extensive structure...`
- Enlarge Figure 7
- Ensure Figure 6 text matches the actual diagram
- Verify the GitHub link works

Where in `paper.tex`:
- wording issue around line 781
- figures and captions around the relevant figure blocks
- code link around lines 622--623

What this addresses:
- R1.1
- R1.7 to R1.9
- overall polish expected by T-BIOM

## What needs to happen

### Highest-priority items

1. Reframe the evaluation more carefully.
   - Do not retreat from the current setup. It is already open-set with respect to training identities.
   - Do not describe it as generic "open-set verification" without qualification.
   - Rename it throughout the paper as `cross-dataset subject-disjoint open-set verification` or `leave-one-dataset-out subject-disjoint verification`.
   - Add a precise definition of what is open-set in your setting:
     - train identities are disjoint from test identities
     - probe/enrol pairs in test are formed from unseen identities
     - impostor trials are comparisons across different unseen identities
     - each held-out image is used once as an enrolment template and compared against all other held-out images
   - Explicitly state what the current setup does not include:
     - a permanently non-enrolled subset of identities in the held-out dataset
     - an explicit unknown-versus-enrolled gallery rejection protocol
   - Clarify in the rebuttal that this stricter gallery-based interpretation is different from the subject-disjoint open-set protocol already used in the paper.

2. Fix the core overclaim in the results section.
   - The manuscript currently claims the method outperforms handcrafted and deep learning methods across all datasets.
   - Table 5 does not support that claim.
   - Replace with a defensible claim such as:
     - best overall cross-dataset generalization on average
     - best or near-best performance on 4/5 target datasets among learned methods
     - stronger robustness under severe domain shift, especially on FV-USM, PolyU, and VERA
   - Explicitly discuss why handcrafted methods are better on FV-300.

3. Strengthen the loss-function justification.
   - Add direct comparison against ArcFace and CosFace at minimum.
   - Add a beta sensitivity study.
   - Explain clearly how the proposed loss differs from standard angular-margin objectives:
     - what the extra angular term is doing
     - why freezing the auxiliary weights after 30 epochs helps
     - why this is useful specifically for cross-dataset or subject-disjoint verification

4. Add real component ablations.
   - Current ablations only tune hyperparameters.
   - Reviewers want to know whether the actual method design matters.
   - Add at least:
     - full model
     - standard convolution stem instead of DSConv stem, while keeping the rest of the model fixed
     - parameter-matched convolutional backbone instead of the Grapher backbone, while keeping the stem and loss fixed
     - optional: disable only the graph operation if this gives a cleaner ablation than swapping the whole backbone

5. Make the protocol fully reproducible.
   - State the exact number of random seeds.
   - State the exact train/validation split rule.
   - State whether all possible genuine/impostor pairs are used.
   - State how many FV-300 samples were removed and how they were identified.
   - Fix the broken code link and provide a working repository or supplementary package.

### Important but secondary items

6. Add practical deployment analysis.
   - report parameter count, model size, FLOPs if possible
   - report inference speed on GPU and CPU
   - briefly discuss accuracy versus complexity trade-off

7. Add stronger analysis for generalization.
   - note that the existing 5-seed runs with 95% confidence intervals already provide evidence of stability
   - add a paired `t`-test against the strongest learned baseline so that comparative claims are also statistically supported
   - feature-space visualization or qualitative explanation
   - explicit discussion of why VERA drops

8. Fix presentation issues.
   - improve Figure 7 readability
   - correct Figure 6 description mismatch
   - fix grammar and wording issues

## Protocol wording to use in the paper

### Recommended wording for the current protocol

Use a paragraph close to the following in Section 3.2:

`In this work, we evaluate under a cross-dataset subject-disjoint open-set verification protocol. For each leave-one-dataset-out split, the model is trained on four datasets and evaluated on the remaining dataset. Because the identities are disjoint across datasets, all test identities are unseen during training. Within the held-out dataset, each image is used in turn as an enrolment template and is compared against every other image from the same dataset using cosine similarity between the corresponding embeddings. Comparisons between images of the same identity form genuine trials, whereas comparisons between images of different identities form impostor trials. Self-comparisons are excluded. Thus, the proposed evaluation is open-set with respect to training identities and simultaneously measures cross-dataset generalization under severe domain shift.`

### One-sentence rebuttal version

`Our original protocol is already open-set in the subject-disjoint verification sense, since all test identities are unseen during training and impostor trials compare unseen probe samples against enrolled templates of different unseen identities.`

### Rebuttal clarification for the stricter reviewer interpretation

Use a short clarification such as:

`We respectfully clarify that the current protocol is already open-set with respect to training identities, since all test identities are unseen during training and the evaluation includes both genuine and impostor verification trials among those unseen identities. We acknowledge that this differs from a stricter fixed-gallery unknown-rejection protocol in which some identities are never enrolled anywhere in the gallery.`

## Experiments to run

### Must-run experiments

1. Loss comparison experiment.
   - Proposed loss vs softmax/CrossEntropy
   - Proposed loss vs ArcFace
   - Proposed loss vs CosFace
   - SphereFace only if easy, otherwise optional

2. Beta ablation for the proposed loss.
   - Example values: 0, 0.1, 0.3, 0.5, 1.0
   - Show where the current value sits and why it was chosen

3. Component ablation.
   - Full model
   - Standard convolution stem instead of DSConv, with the rest fixed
   - Parameter-matched convolutional backbone instead of Grapher, with the rest fixed
   - If compute is tight, do this on one representative split and say so explicitly

4. Baseline fairness check.
   - Reconfirm that all learned baselines used the same training/test protocol
   - Add implementation details for each baseline
   - If any baseline numbers come from original papers, clearly separate that from reproduced results

5. Paired t-test against the strongest learned baseline.
   - Compare OpenVeinNet against the strongest learned baseline, most likely VeinAttNet.
   - Use the same 5 seeds for both methods under each held-out split.
   - Run a paired `t`-test on EER and optionally AUC.
   - Report mean difference and `p`-value for each split.
   - Keep the 95% confidence intervals as the stability measure across seeds.

### Nice-to-have experiments

6. Embedding visualization.
   - t-SNE or UMAP for a few datasets
   - show why the proposed loss or graph backbone improves separation

7. Runtime and complexity table.
   - params, model size, FLOPs, GPU latency, CPU latency

## Comment-by-comment plan

## Editor / EiC / AE

### EiC note

Concern:
- lack of clear motivation for method components
- substantial revisions required
- current work has interest but needs stronger analysis and positioning

How to address:
- narrow the claims
- explicitly motivate each component against finger-vein-specific challenges
- add direct component ablation and stronger loss comparisons
- rewrite discussion to emphasize what is actually demonstrated

### AE note

Concern:
- formulation, experimental design, and conclusions are not yet convincing enough

How to address:
- treat this revision like a near-resubmission, not a cosmetic rebuttal
- the revised manuscript must materially change Sections 1, 3, 4, and the conclusion

## Reviewer 1

### R1.1 Broken code/model link

Concern:
- reproducibility is undermined because the GitHub link does not work

Action:
- create a public working repository before resubmission, or provide anonymized supplementary code/model package if public release is delayed
- include training scripts, config files, seed handling, and pretrained checkpoints
- verify the exact URL in the revised paper

Needs new experiment:
- No

### R1.2 Number of random seeds missing

Concern:
- claims about multiple seeds are not verifiable

Action:
- state exact seed count and seed values
- explain whether confidence intervals are over seeds only, or seeds plus split variation

Needs new experiment:
- Possibly no, if already done and logs exist

### R1.3 Removed FV-300 bad-quality samples not described

Concern:
- data curation is unclear and may affect reproducibility

Action:
- report exact count removed
- define removal criterion
- ideally list sample IDs in supplement or repository
- update Table 3 so counts are internally consistent

Needs new experiment:
- No, unless counts in the paper are wrong and results need rerunning

### R1.4 Training/validation split not given

Concern:
- protocol is incomplete

Action:
- specify split ratio or exact rule
- specify whether split is subject-wise, finger-wise, or image-wise
- specify if it changes across seeds

Needs new experiment:
- No

### R1.5 Missing open-set tailored baseline and missing baseline training details

Concern:
- comparison set may be incomplete or unfair

Action:
- add the Chen et al. 2021 open-set-oriented baseline if feasible
- if not feasible, explain clearly why not and cite it in related work plus limitations
- add a baseline implementation-details table: optimizer, epochs, learning rate, input size, augmentation, embedding dimension, and protocol

Needs new experiment:
- Ideally yes

### R1.6 Ablation scenario not clearly stated

Concern:
- current ablation results are ambiguous

Action:
- state dataset and train/test protocol in every ablation caption and subsection
- ideally use one representative split consistently, and justify why

Needs new experiment:
- Not necessarily, but likely yes because the ablation needs to be expanded anyway

### R1.7 to R1.9 Minor writing and figure issues

Concern:
- grammar, figure mismatch, small plots

Action:
- fix wording on page 5
- align Figure 6 text with actual diagram
- enlarge Figure 7 and simplify legend if needed

Needs new experiment:
- No

## Reviewer 2

### R2.1 Open-set vs cross-dataset confusion

Concern:
- reviewer thinks current protocol may be cross-dataset generalization, not fully convincing open-set verification

Reality:
- your current protocol is already open-set with respect to training identities
- it is subject-disjoint and therefore does include unseen identities
- however, the paper does not define the evaluation rigorously enough
- the reviewer appears to be using a stricter gallery-based interpretation of open-set

Action:
- do not argue that the reviewer is simply wrong
- respond by clarifying terminology and strengthening experiments
- define the current protocol formally with enrolment, probe, genuine, and impostor trial construction
- explicitly state that each held-out image is used once as an enrolment template and compared with all other images from the held-out dataset
- add GAR or GMR at fixed FAR or FMR

Needs new experiment:
- Yes

### R2.2 Fairness of baseline comparison

Concern:
- handcrafted methods outperform some learned baselines, which makes reviewer suspicious of the learned baseline setup

Action:
- add a protocol-equivalence statement for all baselines
- specify whether results were reproduced or quoted
- if reproduced, give implementation details
- explicitly explain why handcrafted methods can remain strong on clean datasets like FV-300 while learned cross-dataset methods may be disadvantaged

Needs new experiment:
- Maybe no, but you need verification and probably reruns if protocol was inconsistent

### R2.3 Novelty justification for DSConv and proposed loss

Concern:
- DSConv appears adapted from prior work
- the proposed loss lacks motivation and comparison to known angular-margin losses

Action:
- cite the original DSConv source clearly
- state that the novelty is not inventing DSConv itself, but adapting and integrating it for tubular finger vein representation under cross-dataset generalization
- compare loss against ArcFace and CosFace
- add beta ablation
- rewrite the intuition around the loss in plain language

Needs new experiment:
- Yes

### R2.4 Complexity and deployment cost

Concern:
- no runtime or cost analysis

Action:
- add parameter count, model size, FLOPs if possible
- add GPU and CPU inference latency
- discuss trade-off against the strongest baseline

Needs new experiment:
- Yes, but lightweight

### R2.5 Generalization analysis and VERA drop

Concern:
- wants significance testing, better explanation of generalization, and discussion of VERA failure mode

Action:
- add paired `t`-testing against the strongest learned baseline using the same 5 seeds
- add a short feature-space or error-analysis figure
- explicitly discuss sensor gap and low image quality for VERA

Needs new experiment:
- Recommended

## Reviewer 3

### R3.1 Limited methodological novelty

Concern:
- reviewer sees the method as a straightforward combination of known parts

Action:
- do not oversell architectural novelty
- shift the contribution statement toward:
  - problem setting and cross-dataset rigor
  - finger-vein-specific adaptation of DSConv plus graph modeling
  - evidence that the combination improves robustness
- add component ablations showing the combination is necessary

Needs new experiment:
- Yes

### R3.2 Loss novelty unclear relative to ArcFace/CosFace/SphereFace

Concern:
- current manuscript does not prove why the proposed loss matters

Action:
- direct comparison to ArcFace and CosFace is mandatory
- add theoretical intuition and practical explanation
- explain whether the method acts like angular regularization rather than a classic margin penalty

Needs new experiment:
- Yes

### R3.3 Claims contradict Table 5

Concern:
- the paper says it outperforms handcrafted and deep methods, but Table 5 shows otherwise on at least one dataset

Action:
- rewrite every global superiority claim
- discuss FV-300 explicitly
- report average rank or mean EER/AUC across target datasets if that helps make the overall picture fairer

Needs new experiment:
- No

### R3.4 Genuine and impostor pair generation unclear

Concern:
- protocol is not reproducible

Action:
- add explicit formulas
- from the counts in Table 4, it appears all ordered non-self pairs are used, not sampling
- write this clearly:
  - genuine trials: all ordered pairs of two different images from the same identity
  - impostor trials: all ordered pairs of images from different identities
- define enrolment and probe selection explicitly: each image is used once as the enrolment template and compared against all other images in the held-out dataset

Needs new experiment:
- No

### R3.5 Cross-dataset EER too high for practical deployment; wants intra-database open-set verification

Concern:
- reviewer sees current numbers as academically interesting but practically weak

Action:
- clarify that the current protocol is already open-set with respect to training identities, but is also intentionally harsher because it combines subject-disjoint testing with cross-dataset shift
- explain that the current evaluation is a stress-test of generalization rather than a fixed-gallery deployment protocol
- position cross-dataset results as a stress-test of generalization, not as the only deployment scenario

Needs new experiment:
- No

### R3.6 Ablation too shallow

Concern:
- current ablations are hyperparameter tuning, not scientific validation

Action:
- extend ablations to include component removal and stronger loss baselines
- state the exact dataset/protocol for all ablations
- add short explanation of why the proposed loss improves representation geometry

Needs new experiment:
- Yes

## Suggested manuscript changes by section

### Title and abstract

- Keep the title, but clarify `cross-dataset subject-disjoint` in the abstract and introduction.
- Replace `consistently outperforms existing techniques` with a more precise summary.

### Introduction

- Define open-set verification carefully.
- Separate:
  - subject-disjoint open-set verification
  - cross-dataset generalization
  - standard intra-database verification
- Tone down novelty claims around DSConv unless you are claiming adaptation, not invention.

### Method

- Cite DSConv origin.
- Rewrite loss explanation for readability.
- Add intuition for the frozen auxiliary weights.

### Experiments

- Add exact protocol details:
  - seed count
  - train/validation split
  - sample removal
  - genuine/impostor pair formulas
  - whether all baselines used identical splits
- Add GAR or GMR at fixed FAR/FMR
- Add runtime/complexity table

### Ablation

- Rename from refinement study to ablation study only if it truly becomes one.
- Add component-level ablations and loss comparisons.

### Discussion / conclusion

- Explicitly discuss FV-300 and VERA
- avoid claiming universal superiority
- emphasize robustness under domain shift, not deployment-readiness across all scenarios

## Best response stance

Use this tone in the rebuttal:

- thank the reviewers for highlighting unclear presentation
- acknowledge that the manuscript did not define protocol details explicitly enough
- clarify that the original setup is already a subject-disjoint, cross-dataset open-set verification setting
- explain clearly why the current pairwise subject-disjoint protocol is already open-set with respect to training identities
- acknowledge that some claims were overstated and revise them
- support novelty with new ablations rather than rhetorical argument

Do not use this tone:

- "the reviewer misunderstood"
- "this is already obvious from the figure"
- "the reviewer is confusing verification with PAD"

Even if true, that will not help. The revision should remove the possibility of confusion.

## Practical meeting checklist for Thursday

1. Decide the exact scope of new experiments.
   - minimum acceptable revision:
     - ArcFace/CosFace comparison
     - component ablation
     - paired `t`-test against the strongest learned baseline
     - runtime table

2. Decide whether to keep the title unchanged.
   - keep it only if you add the standard open-set protocol reviewers expect

3. Verify reproducibility assets.
   - public repo
   - config files
   - pretrained checkpoints
   - dataset split description

4. Rewrite contribution claims before touching the rebuttal.
   - otherwise the same contradictions will remain

5. Assign ownership.
   - experiments
   - protocol documentation
   - figures/tables
   - response letter

## My assessment

This is not a case where a rebuttal alone will work. The paper needs real new experiments plus major rewriting. The good news is that the reviewers are not pulling in three different directions. The central asks are actually coherent:

- define the task better
- distinguish the two meanings of open-set being used
- support the novelty claims better
- make the experiments reproducible
- stop overstating the conclusions

If you address those four points seriously, Reviewer 3 becomes answerable, and Reviewer 1 and Reviewer 2 should become much easier to satisfy.
