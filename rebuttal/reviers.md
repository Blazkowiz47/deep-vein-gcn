We have completed the review process of the above referenced paper for the IEEE Transactions on Biometrics, Behavior, and Identity Science and recommend that your paper undergo a Major Revision.  

Enclosed are your reviews. (If any of the reviews appears to be missing detailed comments, please check if the reviewer uploaded a file on the submission site with comments; those files do not automatically attach to this email.)  When you revise your paper, please prepare a separate document describing how each of the reviewers' comments are responded to in your revision and send it to us by 20-Apr-2026.

When you are ready to submit your revision, visit the following link:
https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-A9BAD691-4113-4B99-91DF-F772C30E307B-70BD1A5B-8E3A-43C4-B07E-2D4CA4B1BCE7-59934/current?idtype=external

When submitting your revised manuscript, you will be able to respond to the comments made by the reviewer(s) in the space provided. You can use this space to document any changes you make to the original manuscript. In order to expedite the processing of the revised manuscript, please be as specific as possible in your response to the reviewer(s)’ questions and comments. You may also upload your responses as separate files for review along with your revision.

When the submission process is complete, you will receive an automated confirmation email immediately. If you did not receive that email, your submission is not yet complete.  

I will contact you should we have any concerns or questions regarding your revision. Otherwise, your revision will be forwarded to the assigned Associate Editor with a request to begin the second round of reviews.

Please be mindful when making your revisions that you still need to maintain the size limitations for papers submitted to IEEE Transactions on Biometrics, Behavior, and Identity Science. Our manuscript types and submission length guidelines (including the main text, the abstract, index terms, illustrations and references) are found at the url below.
Please note that double column will translate more readily into the final publication format.  Our peer review double column templates can also be found at the url below.

https://ieee-biometrics.org/

Text in any color other than black is not acceptable. Your revised paper must include the following:

- Abstract
- Index terms
- Author affiliation information
- Main text
- References
- Figure captions
- Table titles
- Brief biography of each author
(biographies are not required for concise papers or comments papers)

Because this is a revision, we request that you add your author bios and photos at this time. This will help ease the transition to pre-prints if your paper is accepted.

Please do not hesitate in contacting us should you have any questions about our process or are experiencing technical difficulties. You can reach me at p.pundir@ieee.org.

Thank you for your contribution to IEEE Transactions on Biometrics, Behavior, and Identity Science, and we look forward to receiving your revised manuscript.

Thank you,

Mark Nixon
EIC, IEEE Transactions on Biometrics, Behavior, and Identity Science
msn@ecs.soton.ac.uk



**************
Editor Comments

Co-Editor-in-Chief : 1
Comments to the Author:
We have been discussing this paper at the Editorial level. We note that Reviewer 3 noted a lack of clear motivation for the methodological components, suggesting they appear disconnected from the problem. We also note that Reviewer 1 and Reviewer 2 call for substantial revisions. We have gone through the revisions required, but also note the interest expressed by the more supportive reviewers, and even the more negative reviewer's comments on generalisability.  As such there is clearly work of interest here, but there appear to be some major improvements to be made consistent with a top journal like TBIOM. These comments suggest that the work should be analysed more carefully in the light of current literature, whilst making several suggestions to improve the paper. Please follow the excellent comments from the reviewers and the AE when preparing the revised version. We shall look closely at the revised version in the light of the comments made.

Associate Editor: 2
Comments to the Author:
Editorial: According to the review outcome, the reviewers have strong reservation regarding the manuscript. Particularly, they raised critical concern in the formulation, experimental design and the conclusion drawn. Although the topic appears interesting, it is apparent that more than a major revision is necessary. Based on this observation we regret that the manuscript cannot be accepted. While the outcome may be disappointing to you, we hope you find the reviewers' comments useful for further work in this topic.  

********************

Reviewer Comments

Reviewer: 1

Recommendation: Author Should Prepare A Major Revision For A Second Review

Comments:
The manuscript is well written in general and most of the technical details of the approach are given. However, the paper has some shortcomings:

The first but most important one is the release of the source code and pre-trained model. The link https://github.com/Blazkowiz47/OpenVeinNet does not exist. Hence, the reproducibility is not ensured and the results cannot be verified. This definitely needs to be fixed prior to the publication of the paper.
The authors state that each experiment is repeated with multiple random seeds, but the number of runs is not given.
The text for table 3 states that some “bad quality” samples were removed from the FV-300 for the experiments. There is no information on which or how many samples were removed.
The authors state that the samples per identity are deliberately varied between training and validation sets, but the training and validation split is not given.
The authors state that their approach outperforms all the tested approaches from the literature. However, there are some approaches specifically tailored to open-set vein recognition, e.g. [1] which have not been tested by the authors. Furthermore, no details about the training and parameter optimisation of the tested approaches is provided in the paper.
For the ablation study it is not clear which datasets or which open-set scenario has been used. Only for the loss function ablation study the authors state that it was ABDE -> C, but for the number of Grapher blocks and kernel sizes, this information is missing.
There are some minor issues in the paper as well, e.g. on page 5, second column: “The architecture in paper describes a an extensive structure with four initial Grapher blocks, …”
The text for Fig. 6 refers to Block 1 and Pooling, but in the figure there is neither a Block 1 nor a pooling layer.
The Fig. 7 plots are too small and hard to read.


Especially given the non-working link to the source code and the pre-trained network and the missing information for the experimental evaluation, the results are questionable. In the reviewers opinion, a major revision of the paper is necessary to be able to verify the presented results.

References:
[1] Chen, Z., Yu, W., Bai, H., Li, Y. (2021). An Arcloss-Based and Openset-Test-Oriented Finger Vein Recognition System. In: Feng, J., Zhang, J., Liu, M., Fang, Y. (eds) Biometric Recognition. CCBR 2021. Lecture Notes in Computer Science(), vol 12878. Springer, Cham. https://doi.org/10.1007/978-3-030-86608-2_32

Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Good

2.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The authors present a new deep learning based approach for finger vein recognition. The main novelty of the paper is in the combination of CNN that combines dynamic snake convolution with a graph convolutional network and employs a novel loss function called centroid-angular hybrid loss as introduced by the authors. The dynamic snake convolution should be able to effectively capture the tubular structure of the finger vasculature network and the graph convolution should network be able to encode these structures in discriminative feature vectors. The authors test their approach in an open set scenario on five different public finger vein datasets and compare their results with classical feature based as well as deep learning based finger vein recognition approaches. Moreover, the authors claim that the source code and pre-trained models will be released.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Partially

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Lacking in some respects; some cases of interest not tested

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

2.  Does the manuscript contain sufficient and appropriate references?  Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: Chen, Z., Yu, W., Bai, H., Li, Y. (2021). An Arcloss-Based and Openset-Test-Oriented Finger Vein Recognition System. In: Feng, J., Zhang, J., Liu, M., Fang, Y. (eds) Biometric Recognition. CCBR 2021. Lecture Notes in Computer Science(), vol 12878. Springer, Cham. https://doi.org/10.1007/978-3-030-86608-2_32

3.  Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Yes

4.  How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Satisfactory

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Easy to read

6. How is the length of the manuscript?  If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

8. If yes to 7, should it be accepted: After revisions.  Please include explanation under Public Comments below.

Please rate the manuscript overall. Explain your choice.: Good


Reviewer: 2

Recommendation: Author Should Prepare A Major Revision For A Second Review

Comments:
This paper introduces a novel finger vein verification framework named OpenVeinNet, which integrates Dynamic Snake Convolution  with Graph Convolutional Networks for open - set verification scenarios. The proposed approach tackles a crucial challenge in biometric verification, namely generalization to unseen subjects, and exhibits competitive performance across multiple public datasets. The research is technically robust and presents several innovative elements. Nevertheless, several aspects require clarification, enhancement, or additional justification prior to publication in a IEEE TBIOM.
1. The paper asserts to address "open - set verification," yet the described evaluation methodology seems to be cross - dataset evaluation rather than true open - set verification. In true open - set verification, the system must not only handle unseen subjects but also correctly reject impostors who do not belong to any enrolled identity. The current evaluation protocol (training on four datasets and testing on the fifth) assesses cross - domain generalization but does not explicitly test the system's ability to reject unknown identities. It is necessary to clarify whether the evaluation truly constitutes open - set verification or whether it would be more precisely described as cross - dataset closed - set verification. If it is the former, additional experiments demonstrating false acceptance rates at specific false rejection rates would strengthen this claim.
2. The paper compares with seven state - of - the - art methods; however, several concerns regarding the fairness of these comparisons emerge:
MCP, RLT, and WLD are traditional hand - crafted methods, and it is surprising that they outperform many deep learning methods (e.g., ArcVein, LGFIN, FV - ViT) on certain datasets. This raises questions about whether the deep learning baselines were properly implemented and optimized.
Table 5 shows that MCP achieves 99.80% AUC on BCDE→A, while deep learning methods like ArcVein achieve only 89.77%. This significant difference is unusual and requires an explanation—are the deep learning baselines using the same evaluation protocol?
The paper should clarify whether the baseline results were reproduced using the original authors' implementations or taken from published literature and ensure that all methods use identical training/testing splits.
3. Although the paper proposes several novel elements, the claims of contribution need stronger justification:
Dynamic Snake Convolution: The paper states that DSConv "adjusts its sampling locations in a flexible manner that follows the natural contours of the vein patterns." However, DSConv appears to be adapted from existing work (possibly from medical imaging applications). It is necessary to clarify the novelty of DSConv in the context of finger vein recognition and cite the original DSConv work.
Centroid - Angular Hybrid Loss: This loss function combines softmax with an angular component. The mathematical formulation in Equations 2 - 3 is intricate and would benefit from: (a) a clearer explanation of the intuition behind the angular term, (b) ablation studies demonstrating the effect of the β parameter, and (c) comparison with existing margin - based losses (ArcFace, CosFace) used in face recognition.
The paper mentions releasing source code, which is beneficial for reproducibility. Please ensure that the code is properly documented and includes training configurations for all experiments.
4. Architectural Complexity and Computational Cost
The proposed architecture (approximately 4.7 million parameters) is relatively complex, consisting of multiple components (DSConv stem, Grapher blocks, hybrid loss). For practical deployment, especially in resource - constrained environments, the following should be provided: Inference time (FPS) on GPU and CPU, comparison of model size with baseline methods, and a discussion of the computational trade - offs versus accuracy gains.
5. Generalization Analysis
The cross - dataset evaluation is rigorous and praiseworthy. However, the paper could strengthen its analysis by: providing statistical significance tests to determine whether performance differences between OpenVeinNet and the second - best method are statistically significant; discussing the reasons for the better generalization of the method, perhaps through additional analysis of the learned features or visualization of the feature space; and addressing the performance drop on the VERA dataset (89.98% AUC versus >97% on other datasets), which suggests vulnerability to certain types of noise or acquisition conditions.

Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Very Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Good

2.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The proposed approach tackles a crucial challenge in biometric verification, namely generalization to unseen subjects, and exhibits competitive performance across multiple public datasets.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Appears to be - but didn't check completely

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Lacking in some respects; some cases of interest not tested

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

2.  Does the manuscript contain sufficient and appropriate references?  Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: N/A

3.  Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Could be improved

4.  How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Could be improved

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Readable - but requires some effort to understand

6. How is the length of the manuscript?  If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

8. If yes to 7, should it be accepted:

Please rate the manuscript overall. Explain your choice.: Fair


Reviewer: 3

Recommendation: Reject

Comments:
Strengths:
Robust recognition under varying working conditions remains important in biometrics. The paper’s attempt to combine different modeling mechanisms (snake convolution, GNN, and loss optimization) to enhance the generalization and robustness of finger vein recognition is directionally meaningful. Performing cross-dataset evaluation is commendable, as it is a strong indicator of generalization. The paper is generally readable.

However, the current manuscript has the following major weaknesses:
1) The proposed framework essentially combines two existing components: snake convolution for vascular feature extraction and GNN-based relational modeling. The integration appears straightforward, with very limited insights (analytically or experimentally) to the field of finger vein recognition.

2) The proposed loss function is presented as a reformulation of the softmax loss. However, similar angular-margin-based losses (e.g., ArcFace, CosFace, SphereFace) have been extensively studied and widely adopted in biometric recognition tasks. The manuscript does not clearly explain the motivation behind the proposed loss formulation, nor the theoretical or practical advantages over existing angular-margin losses. Without clear justification or comparison with state-of-the-art losses, the novelty and significance of the proposed loss remain unclear.

3) The manuscript repeatedly claims that the proposed method outperforms both handcrafted and deep learning approaches across five datasets. However, the results reported in Table 5 contradict this statement. Specifically, the proposed method performs significantly worse than traditional handcrafted methods such as MCP, RLT, and WLD on certain datasets. This abnormal observation is not discussed or analyzed in the manuscript. Such inconsistencies weaken the credibility of the conclusions.

4) In Table 4 (Section 3.2), the protocol for constructing genuine and impostor pairs is not clearly described, e.g., how the genuine and impostor pairs are generated, whether all possible pairs are used or sampling is performed? The lack of protocol clarity makes it difficult to assess the validity and reproducibility of the reported results.

5) In Table 5 (Section 3.3), the authors report cross-dataset evaluation results, which is indeed a valuable indicator of generalization ability. However, the reported performance under this setting is actually very poor, with EER values ranging from 3% to 18% by the proposed method. Such performance levels are far below the requirements of practical biometric systems. To provide a more meaningful comparison, the authors should also report open-set verification results under the intra-database setting, which is the standard evaluation protocol for vein recognition systems.

6) The ablation experiments in Section 4 (Tables 6–7 and Fig. 8) are limited in scope and depth. The study mainly evaluates different convolution kernel sizes, the number of GrapherBlocks, and three loss variants. However, the datasets used in the ablation experiments are not clearly described, making the results ambiguous. The proposed loss is not compared with strong baseline losses such as ArcFace or CosFace, and there is no analysis explaining why the proposed loss improves performance.

Overall, due to limited methodological novelty, unclear motivation of the proposed loss, insufficient experimental rigor, and several inconsistencies between claims and results, the manuscript does not currently meet the standards expected for publication in IEEE T-BIOM.

Additional Questions:
1. Which category describes this manuscript?: Research/Technology

2. How relevant is this manuscript to the readers of this periodical? If you answer Not very relevant or Irrelevant please explain your rating under Public Comments below.: Very Relevant

1. Please evaluate the significance of the manuscript’s research contribution.: Fair  - Even with the recommended changes, the contribution of this paper is unlikely be significant enough for publication.

2.  Please explain how this manuscript advances this field of research and/or contributes something new to the literature.: The paper proposes a finger vein recognition framework that integrates GNN-based relational modeling with snake convolution and introduces a modified softmax-style loss. While the topic is relevant to biometric recognition, the current manuscript does not reach T-BIOM’s expected level of methodological novelty, clarity of motivation, and experimental rigor/insight. Several key claims appear overstated or inconsistent with the reported results, which weakens confidence in the conclusions.

3. Is the manuscript technically sound? In the Public Comments section, please provide detailed explanations to support your assessment: Partially

4. How thorough is the experimental validation (where appropriate)? Please discuss any shortcomings in the Public Comments section.: Insufficient; clearly inferior to state of the art, or necessary tests are absent

1. Are the title, abstract, and keywords appropriate? If not, please comment in the Public Comments section.: Yes

2.  Does the manuscript contain sufficient and appropriate references?  Please comment and include additional suggested references in the Public Comments section.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.
: NA

3.  Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? If not, please explain your answer in the Public Comments section.: Could be improved

4.  How would you rate the organization of the manuscript? Is it focused? Please elaborate with suggestions for reorganization in the Public Comments section.: Could be improved

5. Please rate the readability of the manuscript. Explain your rating under Public Comments below.: Readable - but requires some effort to understand

6. How is the length of the manuscript?  If changes are suggested, please make explicit recommendations in the Public Comments section.: About right

7. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

8. If yes to 7, should it be accepted:

Please rate the manuscript overall. Explain your choice.: Fair
