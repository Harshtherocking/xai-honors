import torch


def calculate_posterior(
    prior: torch.Tensor,
    likelihood: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the normalised posterior attribution map by combining the
    attention-derived prior and the gradient-based likelihood via a
    Bayesian approximation:

        P(I | Y) = (prior · likelihood) / (|prior| · |likelihood|)

    Args:
        prior      : Tensor ``(B, H, W)`` — EMA-weighted cross-attention map,
                     output of ``calculate_prior()``.
        likelihood : Tensor ``(B, H, W)`` — Grad-CAM saliency map,
                     output of ``compute_gradcam()``.

    Returns:
        Tensor ``(B, H, W)`` of normalised posterior scores in [-1, 1].

    Example
    -------
    >>> posterior = calculate_posterior(prior, likelihood)
    >>> posterior.shape   # (B, H, W)
    """
    if not isinstance(prior, torch.Tensor):
        prior = torch.from_numpy(prior)
    if not isinstance(likelihood, torch.Tensor):
        likelihood = torch.from_numpy(likelihood)

    assert prior.shape == likelihood.shape, (
        f"Shape mismatch: prior {prior.shape} vs likelihood {likelihood.shape}"
    )

    B = prior.shape[0]

    # flatten spatial dims for norming → (B, H*W)
    prior_flat      = prior.view(B, -1)
    likelihood_flat = likelihood.view(B, -1)

    # element-wise product
    numerator = prior_flat * likelihood_flat            # (B, H*W)

    # independent L2 norms with epsilon guard
    eps = 1e-8
    prior_norm      = prior_flat.norm(dim=1, keepdim=True).clamp(min=eps)       # (B, 1)
    likelihood_norm = likelihood_flat.norm(dim=1, keepdim=True).clamp(min=eps)  # (B, 1)

    # normalised posterior
    posterior = numerator / (prior_norm * likelihood_norm)  # (B, H*W)

    # restore spatial shape → (B, H, W)
    return posterior.view_as(prior)