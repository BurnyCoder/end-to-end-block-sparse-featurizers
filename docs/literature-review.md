# Literature and reusable source review

Review completed before implementation/model execution, 2026-10-27. Exact combination searches
did not locate a verified prior output-KL BSF implementation; this is not a novelty claim.

| Source | Reused insight / limitation |
|---|---|
| [Braun et al., 2024](https://arxiv.org/abs/2405.12241) / [Apollo code](https://github.com/ApolloResearch/e2e_sae/tree/83b4add4d2652a4e5c6775a0e2b752897c871a87) | Freeze model, replace activations, optimize output KL. Executable KL weights the original distribution despite ambiguous/reversed prose. Local and downstream fidelity are distinct. |
| [Fel et al., 2026](https://arxiv.org/abs/2606.25234) / [Goodfire code](https://github.com/goodfire-ai/block-sparse-featurizer/tree/219f121ea82d2b19200d1dac918396e6058d7eb9) | DINOv3 final patches, block sparsity, Grassmannian charts and concept maps. Released code differs from paper in gain and QR details. |
| [Revisiting end-to-end SAE training](https://arxiv.org/abs/2503.17272) / [code](https://github.com/adamkarvonen/sae_kl_finetune) | Short KL fine-tuning is a useful existing approach, but this study explicitly uses scratch initialization. |
| [A Deeper Analysis of BSFs](https://arxiv.org/abs/2608.27515) | Splitting/composition and seed instability motivate matched scalar capacity and cautious geometry claims. Tournament TopK is outside this intervention. |
| [SAEBench](https://proceedings.mlr.press/v267/karvonen25a.html) | Fidelity alone does not establish interpretability. |
| [SAE metric audit](https://arxiv.org/abs/2605.18229) | Proxy metrics have limitations; avoid substituting them for the prespecified endpoint. |
| [DINOv3](https://github.com/facebookresearch/dinov3) / [checkpoint](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) | Backbone outputs embeddings, not class probabilities. Official released classification head targets ViT-7B; fit/freeze an explicit ViT-B patch readout. |
| [Imagenette](https://github.com/fastai/imagenette) / [torchvision loader](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Imagenette.html) | Accessible ten-class ImageNet subset; archive verification and class mapping are reused. |
| [Previous workbench](https://github.com/BurnyCoder/block-sparse-featurizer-experiments/tree/6af61910c3ad2ddc4320bff76fe607b6ede13c18) | Reuse extraction recipe and verbatim sanitized logger. Refitting normalization and collecting every dense code are unsuitable here. Prior rabbit scores are not new held-out evidence. |
| [Power analysis](https://aclanthology.org/2020.emnlp-main.745/) / [statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.html) | Pilot-based sample-size estimates are uncertain; training seeds and correlated images are separate sampling units. |
| [ML pitfalls](https://arxiv.org/abs/2108.02497), [Troubling Trends](https://arxiv.org/abs/1807.03341), [NeurIPS checklist](https://neurips.cc/public/guides/PaperChecklist) | Isolate test data, document compute and choices, distinguish evidence from interpretation. |

## Released implementation versus paper

The selected Grassmannian code uses one **unconstrained gain per block** and differentiable QR on
every forward. Appendix D describes a shared positive scalar and occasional projection. We use
the released code's mathematics, computing QR once per forward instead of twice. A parity test
checks outputs/gradients against the vendored reference. No unreported AuxK term is added.

The paper describes max-abs classification pooling but does not release its probe. This project
defines maximum absolute **magnitude**, not signed extreme-value pooling, and states that choice.
Imagenette and the new fixed ten-class probe are deliberate scale/task deviations. The existing
workbench's pretrained rabbit weights are never used to initialize scratch experiments.

## Engineering references

- [PyTorch KL](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html),
  [QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html),
  [autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html),
  [reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
- [uv/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/),
  [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
  [group splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
  [perceptual hashes](https://github.com/JohannesBuchner/imagehash).

No Kaggle dataset or discussion-only implementation was needed: original repositories, published
papers, official model metadata and library documentation supplied the required components.
