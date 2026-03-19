import torch
import torch.nn as nn
import numpy as np
import config


# ── layer resolution ──────────────────────────────────────────────────────────

def _get_last_vision_encoder_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Returns the final layer of the vision encoder for each supported model.

    - BLIP      : model.vision_model.encoder.layers[-1]
    - PaliGemma : model.vision_tower.vision_model.encoder.layers[-1]
    - Qwen3-VL  : model.visual.blocks[-1]
    """
    name = model_name.lower()

    if name == "blip":
        return model.vision_model.encoder.layers[-1]

    if name == "paligemma":
        return model.vision_tower.vision_model.encoder.layers[-1]

    if name == "qwen3-vl":
        return model.visual.blocks[-1]

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Choose from: blip, paligemma, qwen3-vl"
    )


def _get_last_decoder_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Returns the final decoder / lm_head layer for each supported model.

    - BLIP      : model.text_decoder.cls.predictions.decoder
    - PaliGemma : model.lm_head
    - Qwen3-VL  : model.lm_head
    """
    name = model_name.lower()

    if name == "blip":
        return model.text_decoder.cls.predictions.decoder

    if name in ("paligemma", "qwen3-vl"):
        return model.lm_head

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        f"Choose from: blip, paligemma, qwen3-vl"
    )


# ── prompt prefix lengths ─────────────────────────────────────────────────────

# Number of token positions that belong to the input prompt (not generated).
# .backward() is only called from this index onward.
#
# BLIP      : "[CLS] a photograph of [SEP]"  → 5 tokens (indices 0-4)
#             generated tokens start at index 5
# PaliGemma : "describe this image : "       → variable; caller sets start_idx
# Qwen3-VL  : chat template prefix           → variable; caller sets start_idx

PROMPT_PREFIX_LEN = {
    "blip":      4,
    "paligemma": 0,   # override via start_idx if needed
    "qwen3-vl":  0,   # override via start_idx if needed
}


# ── combined hook registration ────────────────────────────────────────────────

def register_grad_hooks(
    model: nn.Module,
    model_name: str | None = None,
    layer_name: str = "vision_encoder_last",
) -> tuple[list[torch.utils.hooks.RemovableHandle], dict, dict]:
    """
    Registers both hooks required for per-token gradient attribution in one call:

    1. **Decoder forward hook** — captures the lm_head / decoder output (kept
       on the computation graph so ``.backward()`` can be called on it).
    2. **Vision encoder backward hook** — collects ``grad_output[0]`` from the
       last vision encoder layer on every ``.backward()`` call.

    Args:
        model      : loaded model (raw or PeftModel).
        model_name : overrides ``config.MODEL`` when provided.
        layer_name : key used in the returned ``gradients`` dict
                     (default ``"vision_encoder_last"``).

    Returns:
        handles        : list of two ``RemovableHandle`` objects.
        decoder_output : dict populated after each forward pass.
                         ``decoder_output["last_layer"]``
                         → Tensor ``(B, seq_len, vocab_size)``, NOT detached.
        gradients      : dict populated after each ``.backward()`` call.
                         ``gradients[layer_name]``
                         → list, one entry per backward pass.
                         ``gradients[layer_name][i][0]``
                         → numpy array ``(B, 577, 1024)`` for BLIP.

    Example
    -------
    >>> handles, decoder_output, gradients = register_grad_hooks(model)
    >>> _ = model(**inputs)
    >>> predictions = decoder_output["last_layer"]   # (B, seq_len, vocab_size)
    >>> processed_grads(predictions, model, processor, gradients)
    >>> for h in handles: h.remove()
    """
    name = (model_name or config.MODEL).lower()

    decoder_output: dict[str, torch.Tensor] = {}
    gradients: dict[str, list]              = {layer_name: []}

    # ── 1. decoder forward hook ───────────────────────────────────────────────
    dec_layer = _get_last_decoder_layer(model, name)

    def decoder_frwd_hook(module: nn.Module, input: tuple, output) -> None:
        decoder_output["last_layer"] = output   # NOT detached

    fwd_handle = dec_layer.register_forward_hook(decoder_frwd_hook)

    # ── 2. vision encoder backward hook ──────────────────────────────────────
    vis_layer = _get_last_vision_encoder_layer(model, name)

    def backward_hook(
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        step_grads = []
        for g in grad_output:
            if g is not None:
                step_grads.append(g.detach().cpu().numpy())
            else:
                step_grads.append(None)
        gradients[layer_name].append(step_grads)

    bwd_handle = vis_layer.register_full_backward_hook(backward_hook)

    return [fwd_handle, bwd_handle], decoder_output, gradients


# ── per-token gradient attribution ───────────────────────────────────────────

def processed_grads(
    predictions: torch.Tensor,
    model: nn.Module,
    processor,
    start_idx: int | None = None,
    verbose: bool = True,
) -> list[int]:
    """
    Calls ``.backward(retain_graph=True)`` on the argmax logit at each token
    position from ``start_idx`` onward, triggering the vision-encoder backward
    hook once per generated token.

    Positions before ``start_idx`` are the prompt prefix (e.g. "[CLS] a
    photograph of [SEP]" for BLIP) and are skipped — gradients for those
    positions are not meaningful for image attribution.

    ``start_idx`` defaults to ``PROMPT_PREFIX_LEN[config.MODEL]``:
        - BLIP      : 5  (skips "[CLS] a photograph of [SEP]")
        - PaliGemma : 0  (no skip by default)
        - Qwen3-VL  : 0  (no skip by default)

    Args:
        predictions : Tensor ``(B, seq_len, vocab_size)`` — the raw decoder
                      output from ``decoder_output["last_layer"]``, still on
                      the computation graph (NOT detached).
        model       : model instance, used for ``model.zero_grad()``.
        processor   : processor / tokenizer used to decode token ids.
        start_idx   : first token position to call ``.backward()`` on.
                      Overrides the default prefix length for the current model.
        verbose     : print each position index and its decoded token text.

    Returns:
        token_ids : list[int] of argmax token ids for positions
                    ``start_idx .. seq_len-1``, in order.

    Accumulates into:
        ``gradients["vision_encoder_last"]``
        After this call the list has ``seq_len - start_idx`` entries.
        Each entry is a list where ``entry[0]`` is a numpy array of shape
        ``(B, 577, 1024)`` for BLIP — the vision-encoder gradient for that
        generated token.

    Example — BLIP
    --------------
    Full token sequence (seq_len = 13):
        idx  0  : [CLS]           ┐
        idx  1  : a               │  prompt prefix — skipped
        idx  2  : photograph      │  (start_idx = 5)
        idx  3  : of              │
        idx  4  : [SEP]           ┘
        idx  5  : a               ┐
        idx  6  : woman           │
        idx  7  : wearing         │  generated tokens — backward called here
        idx  8  : a               │
        idx  9  : blue            │
        idx 10  : and             │
        idx 11  : yellow          │
        idx 12  : sari            │
        idx 13  : [SEP]           ┘

    >>> handles, decoder_output, gradients = register_grad_hooks(model)
    >>> model.train()
    >>> _ = model(pixel_values=pv, input_ids=ids, attention_mask=mask)
    >>> predictions = decoder_output["last_layer"]       # (1, 14, vocab_size)
    >>> token_ids = processed_grads(predictions, model, processor)
    >>> for h in handles: h.remove()
    >>> len(gradients["vision_encoder_last"])             # 9  (indices 5-13)
    >>> gradients["vision_encoder_last"][0][0].shape      # (1, 577, 1024) — 'a'
    >>> gradients["vision_encoder_last"][7][0].shape      # (1, 577, 1024) — 'sari'
    """
    name      = config.MODEL.lower()
    start     = start_idx if start_idx is not None else PROMPT_PREFIX_LEN.get(name, 0)
    seq_len   = predictions.shape[1]

    token_ids: list[int] = []

    for i in range(start, seq_len):
        pred = predictions[0][i]    # (vocab_size,)
        y    = pred.argmax(-1)      # scalar tensor

        token_ids.append(y.item())

        if verbose:
            print(f"Token {i} : {processor.tokenizer.decode(y)} --------")

        # zero_grad BEFORE backward so each token's gradient is isolated
        model.zero_grad()

        # retain_graph=True until the very last position
        keep = i < seq_len - 1
        pred[y].backward(retain_graph=keep)

    return token_ids


# ── CLS tokens to drop per model ─────────────────────────────────────────────
#
# Vision encoders prepend special tokens before the patch sequence.
# These must be stripped before any spatial operation (softmax, reshape, etc.)
#
#   BLIP      : 1 CLS  → grad shape (B, 577, 1024) → drop → (B, 576, 1024)
#               576 = 24 × 24 patches  (384 px image, patch_size=16)
#   PaliGemma : no CLS in SigLIP patch sequence
#               → grad shape (B, 196, 1152) → keep all → (B, 196, 1152)
#               196 = 14 × 14 patches  (224 px image, patch_size=16)
#   Qwen3-VL  : no CLS in native ViT
#               → grad shape (B, 196, 1280) → keep all → (B, 196, 1280)

CLS_TOKENS = {
    "blip":      1,
    "paligemma": 0,
    "qwen3-vl":  0,
}


# ── likelihood calculation ────────────────────────────────────────────────────

def calculate_likelihood(
    gradients: dict,
    layer_name: str = "vision_encoder_last",
    model_name: str | None = None,
) -> np.ndarray:
    """
    Computes P(Y | I) — the likelihood of the generated caption given the image
    — as a spatial saliency map over image patches.

    Implements the four-step pipeline from the paper:

    Step 1 — Drop CLS token(s) and stack per-token gradients
        Raw grad per token: (B, Z+cls, C)  →  drop cls  →  (B, Z, C)
        Stack over T generated tokens       →  (T, B, Z, C)

    Step 2 — ReLU  [eq. grad_relu]
        Suppress negative gradients at the END of accumulation (not per-step):
            ĝ = max(0, ∂Y/∂I)  ∈  R^(Z × C)

    Step 3 — Channel-wise softmax  [eq. grad_softmaxed]
        Normalise each channel c over the spatial dimension Z so that the
        Z patch scores for every channel sum to 1:
            G_z^c = exp(ĝ_z^c) / Σ_{z'} exp(ĝ_{z'}^c)
        After this step each column (fixed c, all z) is a probability distribution.

    Step 4 — Channel importance weights α  [eq. grad_weight]
        Global average pool of the RAW (pre-relu) gradient over Z:
            α_c = (1/Z) Σ_z  ∂Y/∂I_z^c
        Shape: (B, C)

    Step 5 — Weighted aggregation  [eq. the_likelihood]
        P(Y | I) = Σ_c  α_c · G_z^c
        Shape: (B, Z)  — one scalar per image patch.

    Args:
        gradients  : dict returned by ``register_grad_hooks``.
                     ``gradients[layer_name]`` must be populated by
                     ``processed_grads`` before calling this function.
        layer_name : key in the gradients dict (default ``"vision_encoder_last"``).
        model_name : overrides ``config.MODEL`` to select the correct CLS count.

    Returns:
        numpy array ``(B, T, Z)`` — one likelihood map per backward pass
        (generated token), over Z spatial patches.
        T = number of backward passes = number of generated tokens.
        For BLIP  : ``(B, T, 576)``  →  ``likelihood[:, t, :].reshape(24, 24)``
                    to get the spatial map for token t.
        For PaliGemma / Qwen3-VL : ``(B, T, 196)`` → reshape to ``(14, 14)``.

    Example
    -------
    >>> handles, decoder_output, gradients = register_grad_hooks(model)
    >>> model.train()
    >>> _ = model(pixel_values=pv, input_ids=gen_tokens,
    ...           attention_mask=torch.ones_like(gen_tokens))
    >>> processed_grads(decoder_output["last_layer"], model, processor)
    >>> for h in handles: h.remove()
    >>>
    >>> likelihood = calculate_likelihood(gradients)    # (1, T, 576) for BLIP
    >>> saliency_t3 = likelihood[0, 3].reshape(24, 24)  # map for token index 3
    """
    name  = (model_name or config.MODEL).lower()
    n_cls = CLS_TOKENS.get(name, 0)
    store = gradients[layer_name]             # list of T entries

    per_token = []

    for entry in store:
        g = entry[0]                          # (B, Z+cls, C)

        # ── Step 1 : drop CLS token(s) ───────────────────────────────────────
        if n_cls > 0:
            g = g[:, n_cls:, :]              # (B, Z, C)

        # ── Step 2 : ReLU  ĝ = max(0, ∂Y/∂I) ────────────────────────────────
        # Negative gradients suppressed at the end of accumulation, per paper.
        g_hat = np.maximum(0, g)             # (B, Z, C)

        # ── Step 3 : channel-wise softmax over Z ─────────────────────────────
        # Normalise along the patch (Z) axis so each channel sums to 1 over Z.
        g_shift = g_hat - g_hat.max(axis=1, keepdims=True)  # numerical stability
        g_exp   = np.exp(g_shift)
        G       = g_exp / (g_exp.sum(axis=1, keepdims=True) + 1e-8)  # (B, Z, C)

        # ── Step 4 : channel importance weights α = GAP over Z ───────────────
        # Paper uses the RAW gradient (before relu) for α.
        alpha = g.mean(axis=1)               # (B, C)

        # ── Step 5 : weighted sum over channels → (B, Z) ─────────────────────
        token_likelihood = np.einsum("bzc, bc -> bz", G, alpha)  # (B, Z)
        per_token.append(token_likelihood)

    # stack over T tokens → (B, T, Z)
    return np.stack(per_token, axis=1)       # (B, T, Z)


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image
    from utils.load import load_blip
    from utils.inference import generate_caption, prepare_blip_inputs

    model, processor = load_blip()
    model = model.to(config.DEVICE)

    img          = Image.open("images/test/14.jpeg").convert("RGB")
    inputs       = prepare_blip_inputs(img, processor)
    pixel_values = inputs["pixel_values"].to(config.DEVICE)
    input_ids    = inputs["input_ids"].to(config.DEVICE)
    attn_mask    = inputs["attention_mask"].to(config.DEVICE)

    # ── step 1: generate caption cleanly ─────────────────────────────────────
    result     = generate_caption(model, processor, img)
    gen_tokens = result["tokens"].to(config.DEVICE)   # (1, full_seq_len)
    print("Caption :", result["captions"][0])

    # ── step 2: register hooks, teacher-forced forward, per-token backward ───
    handles, decoder_output, gradients = register_grad_hooks(model)
    model.train()

    _ = model(
        pixel_values = pixel_values,
        input_ids    = gen_tokens,
        attention_mask = torch.ones_like(gen_tokens),
    )

    predictions = decoder_output["last_layer"]   # (1, seq_len, vocab_size)
    print(f"predictions shape : {predictions.shape}")

    # For BLIP: start_idx=5 skips [CLS] a photograph of [SEP]
    token_ids = processed_grads(predictions, model, processor)

    for h in handles:
        h.remove()

    # ── step 3: inspect & compute likelihood ─────────────────────────────────
    store = gradients["vision_encoder_last"]
    print(f"\nBackward passes (generated tokens only) : {len(store)}")
    print(f"Grad shape per token                    : {store[0][0].shape}")
    # BLIP @ 384 px → (1, 577, 1024)

    likelihood = calculate_likelihood(gradients)   # (1, T, 576) for BLIP
    print(f"Likelihood map shape : {likelihood.shape}")
    # (B=1, T=num_generated_tokens, Z=576)

    # spatial map for each generated token
    for t, tid in enumerate(token_ids):
        tok_str  = processor.tokenizer.decode(tid)
        grid     = likelihood[0, t].reshape(24, 24)
        grid     = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
        filled   = (grid > 0.5).sum()
        print(f"  token {t:>2} {repr(tok_str):<12}  active patches: {filled}/576")