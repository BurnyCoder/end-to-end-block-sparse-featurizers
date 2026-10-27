"""Local: small CPU tests; global: avoid BLAS oversubscription hiding correctness."""

import torch  # Tiny test QR factorizations need only two CPU threads.

torch.set_num_threads(2)  # The production CLI sets its own runtime policy.
