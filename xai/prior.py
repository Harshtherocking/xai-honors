import torch
import torch.nn as nn
import config


# ── layer resolution ──────────────────────────────────────────────────────────

def _get_decoder_layer(model: nn.Module, model_name: str) -> nn.Module:
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
        f"Unsupported model '{model_name}'. Choose from: blip, paligemma, qwen3-vl"
    )


# ── decoder hook ──────────────────────────────────────────────────────────────

def create_decoder_hook(
    model: nn.Module,
) -> tuple[torch.utils.hooks.RemovableHandle, list]:
    """
    Registers a forward hook on the final decoder / lm_head layer.

    Every forward pass appends the layer output to ``state``, so all steps
    from a ``generate()`` call are preserved in order.

    Args:
        model : loaded model (raw or PeftModel). Model name is read from
                ``config.MODEL``.

    Returns:
        hook_handle : ``RemovableHandle`` — call ``.remove()`` to deregister.
        state       : list of tensors/tuples, one entry per forward pass.

    Example
    -------
    >>> handle, state = create_decoder_hook(model)
    >>> _ = model.generate(...)
    >>> handle.remove()
    >>> # state[i] is the decoder output for the i-th decoding step
    """
    layer = _get_decoder_layer(model, config.MODEL)
    state: list = []

    def _hook(module: nn.Module, input: tuple, output):
        # output may be a tensor or a tuple — store as-is, detached
        out = output.detach().cpu() if isinstance(output, torch.Tensor) else output
        state.append(out)

    handle = layer.register_forward_hook(_hook)
    return handle, state


# ── cross-attention hook ──────────────────────────────────────────────────────

def register_cross_attn_hooks(
    model: nn.Module,
) -> tuple[list[torch.utils.hooks.RemovableHandle], dict[int, list[torch.Tensor]]]:
    """
    Registers forward hooks on every cross-attention layer of BLIP's text
    decoder and accumulates attention probabilities across all forward passes
    (i.e. all decoding steps during ``generate()``).

    Args:
        model : BLIP model (raw or PeftModel).

    Returns:
        handles          : list of ``RemovableHandle`` objects — call
                           ``h.remove()`` on each to deregister all hooks.
        cross_attn_store : dict mapping ``layer_idx`` →
                           ``list[Tensor(B, heads, 1, src_len)]``, one tensor
                           per decoding step. Use ``stack_cross_attn()`` to
                           consolidate into a single tensor per layer.

    Example
    -------
    >>> handles, store = register_cross_attn_hooks(model)
    >>> _ = model.generate(...)
    >>> for h in handles:
    ...     h.remove()
    >>> stacked = stack_cross_attn(store)
    >>> # stacked[0].shape → (B, heads, tgt_len, src_len)
    """
    # { layer_idx : [attn_step_0, attn_step_1, ...] }
    cross_attn_store: dict[int, list[torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook_fn(module: nn.Module, input: tuple, output):
            # output[1] is the attention weight tensor: (B, heads, tgt_len, src_len)
            attn_probs: torch.Tensor = output[1]
            if layer_idx not in cross_attn_store:
                cross_attn_store[layer_idx] = []
            cross_attn_store[layer_idx].append(attn_probs.detach().cpu())
        return hook_fn

    encoder_layers = model.text_decoder.bert.encoder.layer
    for layer_id in range(len(encoder_layers)):
        handle = (
            encoder_layers[layer_id]
            .crossattention.self
            .register_forward_hook(make_hook(layer_id))
        )
        handles.append(handle)

    return handles, cross_attn_store


def stack_cross_attn(
    cross_attn_store: dict[int, list[torch.Tensor]],
) -> list[dict[int, torch.Tensor]]:
    """
    Re-organises cross-attention from layer-first to token-first.

    ``cross_attn_store`` is structured as ``store[layer_idx][token_step]``.
    This function inverts that so the result is indexed as
    ``result[token_step][layer_idx]``.

    Args:
        cross_attn_store : dict returned by ``register_cross_attn_hooks()``,
                           mapping layer_idx → list of per-step tensors.

    Returns:
        list of dicts, one per generated token, where each dict maps
        ``layer_idx`` → ``Tensor(B, heads, 1, src_len)``.

    Example
    -------
    >>> result = stack_cross_attn(store)
    >>> result[0]           # all layer attentions for token 0
    >>> result[0][3]        # layer-3 attention for token 0 → (B, heads, 1, src_len)
    >>> result[5][11]       # layer-11 attention for token 5
    """
    num_steps = len(next(iter(cross_attn_store.values())))

    return [
        {layer_idx: steps[token_step] for layer_idx, steps in cross_attn_store.items()}
        for token_step in range(num_steps)
    ]



def calculate_prior(
    cross: dict[int, list[torch.Tensor]],
    alpha = config.ALPHA
) -> torch.Tensor:
    """
    Aggregates cross-attention into a single tensor.

    Input:
        cross : raw cross_attn_store from register_cross_attn_hooks()
                cross[layer_idx][token_step] -> Tensor(B, heads, tgt_len, src_len)

    Aggregation steps:
        1. stack_cross_attn()      : reorder to token-first
                                     stacked[token_step][layer_idx] -> Tensor(B, heads, tgt_len, src_len)
        2. Mean over heads         : (B, heads, tgt_len, src_len) -> (B, tgt_len, src_len)
        3. Mean over tgt_len       : (B, tgt_len, src_len) -> (B, src_len)
        4. Normalised EMA over layers:
                                     w_l = alpha * (1 - alpha)^(L - l)
                                     w_l = w_l / sum(w_l)   (normalise)
                                     weighted sum -> (B, src_len)
        5. Stack over token steps  : num_tokens x (B, src_len) -> (B, num_tokens, src_len)

    Returns:
        Tensor of shape (B, num_tokens_generated, src_len)
    """
    stacked = stack_cross_attn(cross)
    per_token = []
    alpha = config.ALPHA

    for token_step in stacked:
        # stack all layers: (num_layers, B, heads, tgt_len, src_len)
        layers = torch.stack(list(token_step.values()), dim=0)
        L = layers.shape[0]

        # mean over heads -> (num_layers, B, tgt_len, src_len)
        layers = layers.mean(dim=2)

        # mean over tgt_len -> (num_layers, B, src_len)
        layers = layers.mean(dim=2)

        # compute normalised layer weights: w_l = alpha * (1 - alpha)^(L - l)
        # l is 0-indexed so L - l becomes L-1-l ... L-0 -> exponents: L-1, L-2, ..., 0
        exponents = torch.arange(L - 1, -1, -1, dtype=torch.float32)   # [L-1, L-2, ..., 0]
        weights = alpha * (1 - alpha) ** exponents                      # (num_layers,)
        weights = weights / weights.sum()                               # normalise

        # weighted sum over layers -> (B, src_len)
        # weights: (num_layers,) -> (num_layers, 1, 1) for broadcasting
        weights = weights.view(L, 1, 1).to(layers.device)
        ema = (weights * layers).sum(dim=0)

        # drop CLS token (index 0) from src_len -> (B, src_len - 1)
        ema = ema[:, 1:]

        per_token.append(ema)

    # stack over tokens -> (B, num_tokens, src_len - 1)
    return torch.stack(per_token, dim=1)




# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from PIL import Image
    from utils.load import load_blip, load_paligemma, load_qwenvl_v3
    from utils.inference import generate_caption

    loaders = {
        "blip":      load_blip,
        "paligemma": load_paligemma,
        "qwen3-vl":  load_qwenvl_v3,
    }

    model, processor = loaders[config.MODEL]()
    model = model.to(config.DEVICE)

    img = Image.open("images/test/14.jpeg").convert("RGB")

    # ── decoder hook ──
    dec_handle, dec_state = create_decoder_hook(model)

    result = generate_caption(model, processor, img)

    dec_handle.remove()

    print("Caption         :", result["captions"][0])
    print("Decoder steps   :", len(dec_state))
    print("Decoder step[0] :", type(dec_state[0]),
          getattr(dec_state[0], "shape", None))

    # ── cross-attention hook (BLIP only) ──
    if config.MODEL == "blip":
        ca_handles, ca_store = register_cross_attn_hooks(model)

        generate_caption(model, processor, img)

        for h in ca_handles:
            h.remove()

        stacked = stack_cross_attn(ca_store)

        print(f"\nTotal tokens generated         : {len(stacked)}")
        print(f"Layers per token               : {len(stacked[0])}")
        print(f"Attn shape (token 0, layer 0)  : {stacked[0][0].shape}")  # (B, heads, 1, src_len)