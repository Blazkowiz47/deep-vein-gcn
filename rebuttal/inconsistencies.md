# Remaining Rebuttal Inconsistencies

Date: 2026-05-03

Scope:
- `rebuttal/paper.tex`
- `rebuttal/response_letter.tex`
- `rebuttal/supplementary_material.tex`
- `README.md`
- local code/artifacts used to verify claims

Ignored for this checklist:
- CAH-loss formula simplification, per author decision.
- Local figure/build-path issues, because figures are maintained on Overleaf.

Resolved since the previous checklist:
- Cross-dataset half/full JSONs now have seeds `0,1,2,3` for all five learned methods on all five datasets.
- LGFIN significance `n=20` is now supported by `ablation/significance_tests.json`.
- Chen is no longer in the current significance artifact/table scope.
- Main-paper half/full protocol wording now matches exhaustive pairwise score construction.
- README now includes a checkpoint archive link, but it is access-controlled; see item `Reproducibility 1`.


## Reproducibility / README

1. Checkpoint link appears access-controlled, not public.
   Evidence: `README.md:5`, `README.md:7`, `README.md:9`.
   The SharePoint URL redirects but returns HTTP `401` to an unauthenticated `curl -I -L` request. This is acceptable only if the manuscript/response explicitly frames checkpoints as available via controlled access or contact, not as directly included in the public repository.
