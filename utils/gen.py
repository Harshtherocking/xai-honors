import torch
from PIL import Image
from transformers import AutoProcessor

import config


# ── model-specific initial token IDs ────────────────────────────────────────

def _get_initial_ids(processor: AutoProcessor) -> torch.Tensor:
    """
    Returns the initial generated_ids tensor (1, seq_len).
    Model read from config.MODEL.

    - BLIP      : [CLS]
    - PaliGemma : prompt tokens (from full processor call — same as _prepare_inputs)
    - Qwen3-VL  : prompt tokens from the chat template
    """
    name = config.MODEL.lower()

    if name == "blip":
        return processor.tokenizer(
            "a photograph of", return_tensors="pt", add_special_tokens=True
        ).input_ids                                                 # (1, L)

    if name == "paligemma":
        # must go through the full processor to get the image placeholder tokens
        # — we just return the input_ids length; actual ids come from _prepare_inputs
        raise RuntimeError(
            "PaliGemma: use _prepare_inputs() to get initial ids, not _get_initial_ids()"
        )

    if name == "qwen3-vl":
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe this image."}],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor.tokenizer(
            text_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids                                                 # (1, L)

    raise ValueError(f"Unsupported model '{name}'. Choose from: blip, paligemma, qwen3-vl")


def _get_eos_token_id(processor: AutoProcessor) -> int:
    """
    Returns the EOS / stop token id.
    Model read from config.MODEL.

    - BLIP      : SEP token
    - PaliGemma : EOS token
    - Qwen3-VL  : EOS token
    """
    name = config.MODEL.lower()

    if name == "blip":
        return processor.tokenizer.sep_token_id

    if name in ("paligemma", "qwen3-vl"):
        return processor.tokenizer.eos_token_id

    raise ValueError(f"Unsupported model '{name}'. Choose from: blip, paligemma, qwen3-vl")


# ── input preparation ────────────────────────────────────────────────────────

def _prepare_inputs(image: Image.Image, processor: AutoProcessor) -> dict:
    """
    Returns a dict of all tensors needed for the first forward pass.
    For BLIP  : {"pixel_values": ..., "initial_ids": ...}
    For PaliGemma : full processor output dict (pixel_values + input_ids + ...)
    For Qwen3-VL  : processor output dict (pixel_values + image_grid_thw + input_ids)
    Model read from config.MODEL.
    """
    name = config.MODEL.lower()

    if name == "blip":
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        init_id =  processor.tokenizer(
            "a photograph of", return_tensors="pt", add_special_tokens=True
        ).input_ids
        return {
            "pixel_values": pixel_values,                          # (1, C, H, W)
            # "initial_ids": torch.tensor([[cls_id]], dtype=torch.long),  # (1, 1)
            "initial_ids" : init_id
        }

    if name == "paligemma":
        # image and text MUST be processed together — PaliGemma inserts image
        # placeholder tokens directly into the input_ids sequence.
        inputs = processor(
            text="describe this image : ",
            images=image,
            return_tensors="pt",
        )
        return dict(inputs)   # pixel_values, input_ids, attention_mask, token_type_ids

    if name == "qwen3-vl":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": "Describe this image."},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
        return dict(inputs)   # pixel_values, image_grid_thw, input_ids, attention_mask

    raise ValueError(f"Unsupported model '{name}'.")


# ── single forward pass ──────────────────────────────────────────────────────

def _forward(
    model,
    inputs: dict,
    generated_ids: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """
    Runs one forward pass and returns logits (1, seq_len, vocab_size).
    Model read from config.MODEL.
    """
    name = config.MODEL.lower()

    if name == "blip":
        outputs = model(
            pixel_values=inputs["pixel_values"].to(device),
            input_ids=generated_ids.to(device),
        )
        return outputs.logits

    if name == "paligemma":
        kwargs = {
            "pixel_values": inputs["pixel_values"].to(device),
            "input_ids":    generated_ids.to(device),
        }
        if "token_type_ids" in inputs:
            kwargs["token_type_ids"] = inputs["token_type_ids"].to(device)
        outputs = model(**kwargs)
        return outputs.logits

    if name == "qwen3-vl":
        outputs = model(
            pixel_values=inputs["pixel_values"].to(device),
            image_grid_thw=inputs["image_grid_thw"].to(device),
            input_ids=generated_ids.to(device),
        )
        return outputs.logits

    raise ValueError(f"Unsupported model '{name}'.")


# ── main function ────────────────────────────────────────────────────────────

def generate_caption_logits(
    model,
    processor: AutoProcessor,
    image: Image.Image,
    max_new_tokens: int = config.MAX_TARGET_LENGTH,
    device: str | None = None,
    zero_grad: bool = True,
) -> torch.Tensor:
    """
    Manual greedy decoding for BLIP, PaliGemma, and Qwen3-VL.
    Model is read from config.MODEL.

    Args:
        model         : loaded (Peft)Model
        processor     : matching processor
        image         : PIL.Image.Image
        max_new_tokens: max tokens to generate
        device        : target device (falls back to config.DEVICE)
        zero_grad     : call model.zero_grad() each step (for saliency maps)

    Returns:
        generated_ids : torch.Tensor (1, total_len) including prompt tokens
    """
    name   = config.MODEL.lower()
    device = device or config.DEVICE

    eos_id = _get_eos_token_id(processor)
    inputs = _prepare_inputs(image, processor)

    # set starting generated_ids from prepared inputs
    if name == "blip":
        generated_ids = inputs["initial_ids"]
    else:
        # paligemma / qwen3-vl: prompt input_ids are the starting sequence
        generated_ids = inputs["input_ids"]

    prompt_len = generated_ids.shape[-1]   # used by decode_generated_ids

    model.eval()

    for _ in range(max_new_tokens):
        if zero_grad:
            model.zero_grad()

        logits     = _forward(model, inputs, generated_ids, device)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)        # (1,)

        generated_ids = torch.cat(
            [generated_ids, next_token.unsqueeze(0)], dim=-1
        )

        if next_token.item() == eos_id:
            break

    return generated_ids   # (1, total_len)


# ── decode helper ─────────────────────────────────────────────────────────────

def decode_generated_ids(
    generated_ids: torch.Tensor,
    processor: AutoProcessor,
    prompt_len: int | None = None,
) -> str:
    """
    Strips prompt tokens and decodes to a string.
    Model is read from config.MODEL.

    Args:
        generated_ids : tensor returned by generate_caption_logits
        processor     : matching processor
        prompt_len    : number of prompt tokens to strip; auto-computed if None
    """
    if prompt_len is None:
        name = config.MODEL.lower()
        if name == "blip":
            prompt_len = 1   # just [CLS]
        else:
            # re-run prepare to find prompt length (cheap, no model call)
            prompt_len = _prepare_inputs.__wrapped__ if hasattr(_prepare_inputs, "__wrapped__") else None
            # fallback: decode everything and let skip_special_tokens clean up
            prompt_len = 0

    caption_ids = generated_ids[:, prompt_len:]
    return processor.tokenizer.batch_decode(caption_ids, skip_special_tokens=True)[0]


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.load import load_blip, load_paligemma, load_qwenvl_v3

    loaders = {
        "blip":      load_blip,
        "paligemma": load_paligemma,
        "qwen3-vl":  load_qwenvl_v3,
    }

    model, processor = loaders[config.MODEL]()
    model = model.to(config.DEVICE)

    img = Image.open("images/test/14.jpeg").convert("RGB")

    gen_ids = generate_caption_logits(model, processor, img)
    caption = decode_generated_ids(gen_ids, processor)

    print("Generated IDs :", gen_ids)
    print("Caption       :", caption)