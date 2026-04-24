# Must-run new experiments


1. True open-set verification experiment with non-enrolled identities Your current leave-one-dataset-out protocol evaluates unseen test identities, but all probe identities appear to come from the enrolled test population. Reviewer 2 and Reviewer 3 are effectively asking for a stricter protocol where some probe subjects are not enrolled at all and must be rejected using a fixed threshold.

Run this for each leave-one-dataset-out case:

· split test identities into enrolled and non-enrolled unknown

· use a threshold fixed on validation/dev data

· report acceptance of enrolled probes and rejection of non-enrolled probes

2. Intra-database open-set verification experiment Reviewer 3 explicitly asked for a more standard within-database setting in addition to cross-dataset testing. Your paper currently focuses on leave-one-dataset-out cross-dataset verification.

Run, for each dataset or at least the main datasets:

· train / val / test on disjoint identities within the same dataset

· create enrolled vs unknown probe identities

· report open-set verification performance

3. Loss comparison against strong angular-margin baselines Your current loss discussion introduces the Centroid Angular Hybrid Loss, but the reviewers want direct comparison against ArcFace / CosFace / SphereFace-type strong baselines, not just weaker alternatives. This comparison is essential.

Run:

· CE / Softmax

· ArcFace

· CosFace

· optionally SphereFace

· your CAH loss



4. Beta sensitivity study for the proposed loss Reviewer 2 explicitly asked for the effect of the balancing parameter 𝛽. Since 𝛽 appears in your loss formulation, this is a direct missing experiment.

Run a sweep such as:

· 𝛽=0,0.1,0.25,0.5,1.0 and report AUC / EER.

5. Component ablation isolating DSConv and graph modelling Reviewer 3’s main criticism is that the architecture looks like a straightforward combination of existing pieces. To answer that, you need ablations that isolate the contribution of each part. Your current manuscript motivates DSConv and Grapher blocks, but the reviewers want stronger evidence that each is needed. Run:

· standard conv stem + no graph

· DSConv stem + no graph

· standard conv stem + graph backbone

· DSConv stem + graph backbone

· full model

6. Backbone depth / kernel / graph-size ablation with clearly stated protocol The paper already includes some ablation content, but Reviewer 1 and Reviewer 3 say the dataset/scenario is unclear. Your manuscript also contains architectural inconsistency around the Grapher block description, so a clean ablation table is needed. Run clearly on one declared protocol, for example:

· ABDE→C

· BCDE→A and vary:

· number of grapher units / stages

· k-nearest neighbours

· stem kernel sizes





Strongly recommended new experiments

7. Statistical significance testing Reviewer 2 asked for significance testing between proposed and second-best method. Since paper already reports mean and 95% CI over seeds but not formal pairwise significance, add this.

Run:

· paired significance tests across seeds / splits

· especially against VeinAttNet and best handcrafted baselines

8. Analysis of the BCDE→A anomaly where handcrafted methods beat many deep models (Check if this is possible) Table 5 in manuscript shows MCP / RLT / WLD doing unexpectedly well on some cases, while the text says your method consistently outperforms handcrafted and deep methods. That contradiction is one reason the reviewers lost confidence. Run targeted analysis such as:

· performance by dataset quality level

· vessel enhancement or image-quality subgroup analysis

· same evaluation protocol for all methods if you reimplement them

9. Analysis of the VERA performance drop Reviewer 2 asked for discussion of why the VERA case is much weaker. Your paper itself notes that VERA is the most difficult and lowest-quality dataset.

Useful experiments:

· quality/noise robustness study

· simple augmentation/domain-robustness study

· error analysis on VERA

10. Feature-space visualisation Reviewer 2 asked for better explanation of why the method generalises. A feature-space analysis can help support the DSConv + graph + loss story. Since your paper already argues compactness and separation in the embedding space, this is aligned with the manuscript.

Run:

· t-SNE / UMAP for CE vs ArcFace vs CAH loss

· maybe class-centroid angle distributions

11. Threshold-based operating-point evaluation Reviewer 2 asked for false acceptance at specific false rejection operating points. Since your manuscript currently reports mainly AUC and EER, add thresholded operating points.

Report, for both current protocol and stricter open-set protocol:

· TAR @ FAR=1%

· TAR @ FAR=0.1% if stable

· FRR / FMR under fixed threshold

· unknown acceptance rate for non-enrolled probes

I know we discussed this before, but results are not encouraging, so check if this can be added and it is must as reviewer is already asking for this. Just check threshold under which results are stable so that we can add it.


12. Runtime and deployment cost experiment Reviewer 2 explicitly asked for inference time and computational trade-offs. Your manuscript gives parameter count and hardware, but not speed.

Run:

· GPU inference time per image

· CPU inference time per image

· throughput / FPS

· model size comparison with selected baselines



NOTE: for all ablation experiment use one set of dataset and evaluation protocol and keep this constant for all ablation study.