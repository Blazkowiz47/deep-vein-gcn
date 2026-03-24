# Response to Reviewers

## Manuscript

**Title:** OpenVeinNet: A Novel Open-Set Finger Vein Verification Network Using Multi-Graph Learning and Centroid-Angular Hybrid Loss

**Manuscript ID:** [ADD MANUSCRIPT ID]

## Cover Note to the Editor

We thank the Editor-in-Chief, Associate Editor, and all reviewers for their careful reading and constructive feedback. We have revised the manuscript substantially to address the concerns regarding problem formulation, methodological motivation, experimental protocol, reproducibility, and claim clarity.

The revised manuscript includes the following major changes:

1. We clarified that the original evaluation is already open-set with respect to training identities and now define it explicitly as `cross-dataset subject-disjoint open-set verification`.
2. To address the stricter enrollment-based interpretation of open-set verification, we additionally added [an enrollment-based intra-database open-set protocol / new verification experiments under a fixed-gallery setting] and now report operating-point metrics such as [GAR/GMR] at fixed [FAR/FMR].
3. We expanded the ablation study to include direct architectural component analysis and comparison with strong margin-based losses such as [ArcFace] and [CosFace].
4. We revised the claims throughout the manuscript to ensure that all conclusions are fully consistent with the reported results.
5. We added missing reproducibility details, including the exact number of seeds, training/validation split, pair-construction protocol, data curation details, and implementation details for all reproduced baselines.
6. We added computational-cost analysis including [parameter count / model size / FLOPs / GPU latency / CPU latency].
7. We fixed the source-code release details and updated the repository link to: [ADD WORKING URL].

We respond to each comment in detail below. Reviewer comments are reproduced in **bold**, followed by our responses.

## Summary of Major Revisions

### Problem formulation and evaluation

- Added a formal definition of the evaluation protocol in Section [X].
- Clarified that the original protocol is already open-set with respect to training identities and renamed it to `cross-dataset subject-disjoint open-set verification` throughout the manuscript.
- Added [an enrollment-based intra-database open-set evaluation / a fixed-gallery unknown-rejection protocol] in Section [X], Table [X], and Figure [X].
- Added [GAR/GMR] at fixed [FAR/FMR] operating points.

### Methodological motivation and novelty support

- Clarified that DSConv is adapted from prior work and cited the original source in Section [X].
- Revised the contribution statements to avoid overstating architectural novelty.
- Added direct loss comparisons against [Softmax / ArcFace / CosFace / optional SphereFace].
- Added a sensitivity analysis for the hyperparameter `beta`.
- Added component ablations for the DSConv stem and graph backbone.

### Reproducibility and protocol details

- Added the exact number of random seeds: [N].
- Added the exact seed values: [LIST OR OMIT IF NOT NEEDED].
- Added the training/validation split protocol in Section [X].
- Added the FV-300 sample-removal criterion and exact number of removed samples in Section [X] / Table [X].
- Added exact genuine/impostor pair-generation rules in Section [X].
- Added implementation and optimization details for reproduced baselines in Section [X] / supplementary material.

### Presentation and analysis

- Revised the discussion of FV-300 and VERA performance in Section [X].
- Clarified that the reported 95% confidence intervals are computed over [five] random seeds and therefore reflect training stability; [if added: we also included a formal pairwise significance test against the strongest learned baseline].
- Enlarged and improved readability of Figure 7.
- Corrected Figure 6 description and minor textual errors.

---

## Response to the Editor-in-Chief

**We note that Reviewer 3 observed a lack of clear motivation for the methodological components, and Reviewers 1 and 2 also requested substantial revisions. The work appears interesting, but major improvements are needed consistent with a top journal like T-BIOM.**

Response:

We appreciate this assessment and agree that the original manuscript did not make the motivation and scope sufficiently explicit. In response, we substantially revised the manuscript in four directions:

1. We clarified the problem formulation and now precisely define the original evaluation as `cross-dataset subject-disjoint open-set verification`, while also adding [an enrollment-based intra-database open-set protocol / complementary fixed-gallery experiments].
2. We strengthened the methodological justification by adding direct component ablations and comparisons against established angular-margin losses.
3. We revised the claims throughout the manuscript so that they align exactly with the quantitative evidence.
4. We expanded the reproducibility and protocol details, including pair construction, train/validation splits, seed counts, data curation, and baseline implementation details.

We believe these changes materially improve the rigor, clarity, and scientific positioning of the work.

## Response to the Associate Editor

**The reviewers raised critical concerns in the formulation, experimental design, and conclusions drawn.**

Response:

We agree that the original manuscript required a substantial revision in these areas. Accordingly, we have:

- reformulated the evaluation section for precision and reproducibility,
- added new experiments requested by the reviewers,
- expanded the ablation study,
- softened overstated claims,
- added practical and statistical analyses, and
- clarified the scientific role of each methodological component.

We hope the revised manuscript now addresses the concerns regarding formulation, evaluation rigor, and conclusion validity.

---

## Reviewer 1

### Comment 1

**The source code and pre-trained model link does not exist. Reproducibility is therefore not ensured.**

Response:

We thank the reviewer for highlighting this important issue. We agree that the broken repository link undermined reproducibility. In the revised manuscript, we have corrected the repository link and made the code and pretrained models available at:

`[ADD URL]`

The repository now includes:

- training scripts,
- inference scripts,
- model configuration files,
- seed handling,
- pretrained checkpoints, and
- instructions to reproduce the main experiments.

Changes in manuscript:

- Updated the code-release statement in Section [X].
- Added repository details in [supplementary / appendix / footnote].

### Comment 2

**The manuscript states that each experiment is repeated with multiple random seeds, but the number of runs is not given.**

Response:

We agree that this detail should have been reported explicitly. In the revised manuscript, we now specify that each experiment was repeated with `[N]` random seeds, namely `[SEEDS]`. We also clarified whether the reported confidence intervals were computed over seed variation only or over both seed and split variation.

Changes in manuscript:

- Added seed count and evaluation protocol details in Section [X].

### Comment 3

**Table 3 states that some “bad quality” samples were removed from FV-300, but there is no information on which or how many samples were removed.**

Response:

We thank the reviewer for pointing out this omission. We now explicitly report:

- the exact number of removed FV-300 samples: `[COUNT]`,
- the criterion used for removal: `[CRITERION]`, and
- [optionally] the sample identifiers in the supplementary material / repository.

This information has been added to improve transparency and reproducibility.

Changes in manuscript:

- Updated Table [X].
- Added sample-removal description in Section [X].

### Comment 4

**The training and validation split is not given.**

Response:

We agree and have now specified the training/validation split procedure clearly. The revised manuscript states:

- split ratio or exact rule: `[RULE]`,
- whether the split is subject-wise, finger-wise, or image-wise: `[TYPE]`,
- whether the split varies across seeds: `[YES/NO, DESCRIPTION]`.

Changes in manuscript:

- Added split protocol in Section [X].

### Comment 5

**The manuscript does not evaluate some approaches specifically tailored to open-set vein recognition, e.g. Chen et al. (2021). In addition, training and parameter optimization details for the tested approaches are not provided.**

Response:

We thank the reviewer for this suggestion. In the revised manuscript, we have [added / discussed] the open-set-oriented method of Chen et al. [REF]. Specifically:

- [If implemented:] We included this method in the comparison under the same protocol as the other baselines.
- [If not implemented:] We added a discussion in the revised manuscript acknowledging this relevant work and clarifying the reason it was not included experimentally, namely `[REASON]`.

In addition, we now provide implementation and optimization details for all reproduced baselines, including:

- optimizer,
- learning rate,
- number of epochs,
- input size,
- augmentation,
- loss function,
- embedding dimension, and
- protocol consistency with the proposed method.

Changes in manuscript:

- Added baseline-implementation details in Section [X] / Table [X].
- Added discussion of Chen et al. [REF] in Section [X].

### Comment 6

**For the ablation study it is not clear which datasets or open-set scenario were used.**

Response:

We agree. In the revised manuscript, we now explicitly state the exact dataset and protocol used for each ablation experiment in the text, table captions, and figure captions. We also expanded the ablation study to include direct component-level analysis rather than only hyperparameter refinement.

Changes in manuscript:

- Revised Section [X].
- Updated Table [X], Table [X], and Figure [X] captions.

### Comment 7

**There are minor issues in the paper, including wording on page 5, Figure 6 description mismatch, and Figure 7 plots being too small.**

Response:

We thank the reviewer for these careful observations. We corrected the wording issue, aligned the Figure 6 description with the actual architecture, and enlarged Figure 7 to improve readability.

Changes in manuscript:

- Corrected wording in Section [X].
- Revised Figure 6 caption/text.
- Improved Figure 7 layout and readability.

---

## Reviewer 2

### Comment 1

**The paper claims “open-set verification,” but the evaluation appears to be cross-dataset evaluation rather than true open-set verification. Additional experiments demonstrating false acceptance rates at specific false rejection rates would strengthen this claim.**

Response:

We appreciate this important comment. We respectfully clarify that our original protocol is already open-set in the subject-disjoint verification sense. Specifically, the model is trained on four datasets and evaluated on the fifth, and because the identities are disjoint across datasets, all test identities are unseen during training. Within the held-out dataset, each image is used in turn as an enrolment template and is compared against every other image from that dataset; same-identity comparisons form genuine trials and different-identity comparisons form impostor trials.

We agree, however, that the original manuscript did not define this setting precisely enough, and that it also did not clearly distinguish this `subject-disjoint open-set` protocol from the stricter `fixed enrolled gallery with non-enrolled unknown probes` interpretation of open-set verification that the reviewer appears to have in mind.

To remove ambiguity, we made the following changes:

1. We now refer to the original setting as `cross-dataset subject-disjoint open-set verification`.
2. We formally define enrolment, probe, genuine, and impostor trials in Section [X], including the fact that each held-out image is used once as the enrolment template and compared against all other held-out images.
3. To address the stricter operational interpretation of open-set verification, we additionally added [an enrollment-based intra-database open-set evaluation / a fixed-gallery unknown-rejection experiment] in Section [X], Table [X], and Figure [X].
4. We now report [GAR/GMR] at fixed [FAR/FMR] operating points in addition to AUC and EER.

We believe these revisions address the terminology issue while also broadening the evaluation in the way suggested by the reviewer.

Changes in manuscript:

- Revised terminology throughout the manuscript.
- Added formal protocol definition in Section [X].
- Added [enrollment-based intra-database open-set] experiments in Section [X].

### Comment 2

**Handcrafted methods outperform several deep learning baselines on certain datasets. This raises questions about fairness of comparison and whether all methods use the same protocol.**

Response:

We agree that this behavior deserves explicit discussion. In the revised manuscript, we now clarify that all reproduced baselines were evaluated under the same training/testing protocol as the proposed method. We also added the implementation details needed to make this comparison auditable.

In addition, we expanded the discussion to explain why handcrafted methods can remain competitive, especially on cleaner datasets such as FV-300. In particular, handcrafted vein-enhancement methods can perform strongly when the imaging conditions are favorable and the vein structures are high-contrast, whereas learned cross-dataset models can be more sensitive to domain shift when trained across heterogeneous datasets.

Changes in manuscript:

- Added protocol-consistency statement in Section [X].
- Added baseline implementation details in Section [X].
- Added discussion of FV-300 behavior in Section [X].

### Comment 3

**DSConv appears adapted from existing work, and the proposed loss requires clearer motivation, beta ablation, and comparison with ArcFace/CosFace.**

Response:

We thank the reviewer for this constructive suggestion. We have revised the manuscript to make the methodological positioning more precise.

For DSConv:

- We now explicitly cite the original DSConv work.
- We clarify that our contribution is not the invention of DSConv itself, but its adaptation and integration into a finger-vein verification pipeline designed for subject-disjoint and cross-dataset generalization.

For the proposed loss:

- We rewrote the intuition behind the angular term in clearer terms.
- We added a beta sensitivity analysis in Section [X] / Table [X] / Figure [X].
- We added direct comparisons against [ArcFace] and [CosFace] in Section [X] / Table [X].

These additions make the motivation and empirical role of the proposed loss substantially clearer.

Changes in manuscript:

- Added DSConv citation and clarification in Section [X].
- Revised loss explanation in Section [X].
- Added new ablation/comparison results in Section [X].

### Comment 4

**Please provide inference time, model size, and computational trade-off discussion.**

Response:

We agree that practical deployment considerations are important. In the revised manuscript, we added a computational analysis including:

- parameter count,
- model size,
- [FLOPs / MACs],
- GPU inference latency, and
- CPU inference latency.

We also added a short discussion of the trade-off between computational cost and verification performance relative to the strongest competing learned baselines.

Changes in manuscript:

- Added computational analysis in Section [X] / Table [X].

### Comment 5

**The paper would benefit from statistical significance testing, analysis of why generalization is better, and discussion of the performance drop on VERA.**

Response:

We thank the reviewer for these suggestions. In the revised manuscript, we now make explicit that the reported 95% confidence intervals are computed over [five] independent random-seed runs and therefore quantify the stability and uncertainty of the reported performance. We expanded the discussion of why the proposed architecture generalizes better, and we added a dedicated discussion of the VERA performance drop, including the impact of sensor differences, image quality, and acquisition variability.

[If a formal test was added:]
In addition, we included [a paired bootstrap / non-parametric significance test] against the strongest competing learned baseline.

[Optional:]
We also added [feature-space visualization / error analysis] to illustrate the improved separation achieved by the proposed method.

Changes in manuscript:

- Added analysis in Section [X].
- Added [new figure / table] in Section [X].

---

## Reviewer 3

### Comment 1

**The framework mainly combines existing components, and the integration appears straightforward with limited insight.**

Response:

We appreciate this frank assessment. We agree that the original manuscript did not provide enough evidence to justify the benefit of combining these components. In the revised version, we therefore avoid overstating architectural novelty and instead focus on demonstrating the practical and scientific role of each component more clearly.

Specifically, we added component-level ablations comparing:

- [standard convolution stem + graph backbone],
- [DSConv stem + simpler backbone],
- [full model], and
- [other feasible variants].

These results show the contribution of each component to the final performance and support the design choices more directly than the original manuscript.

Changes in manuscript:

- Revised contribution statements in Section [X].
- Added component ablations in Section [X].

### Comment 2

**The proposed loss lacks clear motivation and is not compared with ArcFace/CosFace/SphereFace.**

Response:

We thank the reviewer for this important point. We have substantially revised the loss-related part of the manuscript.

First, we rewrote the explanation of the proposed loss to better clarify its motivation and how it differs from standard angular-margin formulations. In particular, we now explain that the additional angular term is intended to encourage more stable and discriminative embedding geometry under subject-disjoint verification settings.

Second, we added direct empirical comparison against strong margin-based losses, including:

- [ArcFace],
- [CosFace], and
- [SphereFace, if included].

Third, we added a sensitivity analysis for the hyperparameter `beta`.

These additions provide both conceptual and empirical support for the proposed loss.

Changes in manuscript:

- Revised Section [X].
- Added Table [X] / Figure [X].

### Comment 3

**The manuscript claims the proposed method outperforms handcrafted and deep learning approaches across five datasets, but Table 5 contradicts this.**

Response:

We agree with the reviewer. The wording in the original manuscript was too strong and did not accurately reflect the per-dataset results. We have revised the claims throughout the abstract, introduction, results, and conclusion so that they are fully consistent with Table [X].

In particular, we no longer claim universal superiority over all handcrafted and deep learning methods on every target dataset. Instead, we now state more precisely that the proposed method demonstrates strong overall cross-dataset generalization and achieves the best or near-best performance among learned methods across the evaluated conditions, while handcrafted methods remain strong on certain cleaner datasets such as FV-300.

We also added explicit discussion of this behavior in the results section.

Changes in manuscript:

- Revised claims in Abstract, Introduction, Results, and Conclusion.
- Added discussion of FV-300 in Section [X].

### Comment 4

**The protocol for constructing genuine and impostor pairs is not clearly described.**

Response:

We agree and thank the reviewer for identifying this gap. We have now added a precise description of pair construction in Section [X].

Specifically, we define:

- genuine trials as all ordered pairs `(I_i^a, I_i^b)` with `a != b`, where both images belong to the same identity `i`,
- impostor trials as all ordered pairs `(I_i^a, I_j^b)` with `i != j`, where the two images belong to different identities,
- each image in the held-out dataset is used once as the enrolment template and compared against all other held-out images,
- self-comparisons are excluded, and no pair subsampling is performed,
- and the counts in Table [X] are obtained directly from these all-pairs constructions.

This revision makes the evaluation protocol fully reproducible and directly addresses the ambiguity noted by the reviewer.

Changes in manuscript:

- Added pair-construction details in Section [X].
- Updated Table [X] caption and surrounding text.

### Comment 5

**Cross-dataset performance is too poor for practical biometric systems, and the paper should also report open-set verification results under the intra-database setting.**

Response:

We appreciate this comment. We respectfully clarify that the original protocol is already open-set with respect to training identities, since all identities in the held-out dataset are unseen during training and the evaluation includes both genuine and impostor verification trials among those unseen subjects. Our intention was to evaluate robustness under severe domain shift, which is considerably more challenging than standard intra-database verification.

At the same time, we agree that a more deployment-oriented evaluation with a fixed enrolled gallery and non-enrolled unknown probes is useful for practical interpretation. We therefore added [an enrollment-based intra-database open-set protocol / complementary fixed-gallery experiments] for all five datasets and now report:

- AUC,
- EER, and
- [GAR/GMR] at fixed [FAR/FMR].

We also revised the manuscript text to position the original cross-dataset results as a stress-test of generalization rather than as the only deployment-relevant scenario.

Changes in manuscript:

- Added new experiments in Section [X].
- Revised discussion in Section [X] and Conclusion.

### Comment 6

**The ablation study is limited in scope and does not compare the proposed loss with strong baseline losses such as ArcFace or CosFace.**

Response:

We agree. The original ablation section functioned more as a refinement study than as a full ablation analysis. In the revised manuscript, we expanded this section substantially.

The revised ablation section now includes:

- explicit dataset/protocol identification for each experiment,
- component-level architectural ablations,
- comparison with [ArcFace] and [CosFace],
- beta sensitivity analysis, and
- revised discussion explaining why the proposed loss improves the learned representation.

We believe this addresses the reviewer’s concern regarding depth and scientific insight.

Changes in manuscript:

- Reworked Section [X].
- Added new tables/figures [X], [X], [X].

---

## Additional Global Changes Not Tied to a Single Comment

- Revised the abstract to avoid overstated claims.
- Revised the introduction to define the task more precisely.
- Clarified the role of DSConv and graph modeling in relation to finger-vein-specific structure.
- Improved figure readability and corrected textual inconsistencies.
- Added author bios/photos and all required revision-time submission elements.

## Closing Note

We thank the Editor and reviewers again for their constructive comments. We believe the revised manuscript has been significantly strengthened in terms of clarity, rigor, reproducibility, and scientific positioning, and we hope it is now suitable for further consideration in IEEE T-BIOM.
