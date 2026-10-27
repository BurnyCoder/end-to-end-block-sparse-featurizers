"""Global: the research intervention. Local: explicit KL orientation and loss dispatch.

Adapted from ApolloResearch/e2e_sae@83b4add4d2652a4e5c6775a0e2b752897c871a87,
e2e_sae/losses.py; https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html.
"""

import torch.nn.functional as F  # Stable log_softmax and the documented KL convention.


def forward_kl(student_logits, teacher_logits):
    """Local: return per-image KL(p_teacher || q_student); global: distill fixed behavior."""
    student_logp = F.log_softmax(student_logits.float(), dim=-1)  # Accumulate in float32.
    teacher_logp = F.log_softmax(teacher_logits.detach().float(), dim=-1)  # No teacher gradients.
    return F.kl_div(student_logp, teacher_logp, log_target=True, reduction="none").sum(-1)


def objective(name, kl, mse):
    """Local: no implicit penalties; global: isolate objective choice in the comparison."""
    if name == "kl":  # Primary intervention is genuinely KL-only.
        return kl
    if name == "mse":  # Local reconstruction baseline.
        return mse
    if name == "kl_mse":  # Explicit predeclared regularization ablation.
        return kl + 0.1 * mse
    raise ValueError(f"Unknown objective: {name}")  # Typos cannot silently change the study.
