# Reused source

- `goodfire/base.py`, `goodfire/grassmannian.py` and their LICENSE are verbatim from
  [Goodfire revision 219f121](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9).
  The executable adapter in `src/e2e_bsf/featurizer.py` retains this mathematical implementation.
- `src/e2e_bsf/logging_utils.py` is reused verbatim from the user's own
  [workbench revision 6af6191](https://github.com/BurnyCoder/block-sparse-featurizer-experiments/blob/6af61910c3ad2ddc4320bff76fe607b6ede13c18/src/bsf_experiments/logging_utils.py).
- The KL formulation is independently expressed using the documented PyTorch API, with
  attribution to [Apollo's loss implementation](https://github.com/ApolloResearch/e2e_sae/blob/83b4add4d2652a4e5c6775a0e2b752897c871a87/e2e_sae/losses.py).
  No Apollo source file is copied. DINO weights have separate upstream terms and are not redistributed.
