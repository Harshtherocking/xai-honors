import torch
from dataclasses import dataclass, field
from PIL import Image
from transformers import AutoProcessor
import config


from undecorated import undecorated
from types import MethodType






@dataclass
class DecodeConfig:
    do_sample: bool = False         # False → greedy / beam; True → stochastic
    temperature: float | None = None   # <1 sharper, >1 more random
    top_k: int | None = None           # keep only top-K logits
    top_p: float | None = None         # nucleus: smallest set summing to p
    num_beams: int = 1                 # 1 = no beam search
    early_stopping: bool = False       # stop beams when all hit EOS
    repetition_penalty: float = 1.0   # >1 discourages repeated tokens
    length_penalty: float = 1.0       # >1 favours longer outputs (beam only)
    max_new_tokens: int = config.MAX_TARGET_LENGTH

    def to_generate_kwargs(self) -> dict:
        kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "repetition_penalty": self.repetition_penalty,
            "length_penalty": self.length_penalty,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        return kwargs





# input preparing
def prepare_blip_inputs(image: Image.Image, processor: AutoProcessor) -> dict:
    text = "a photograph of"
    inputs = processor(images=image, text = text, return_tensors="pt")

    return dict(inputs)


def prepare_paligemma_inputs(image: Image.Image, processor: AutoProcessor) -> dict:
    inputs = processor( text="describe this image : ", images=image, return_tensors="pt")
    return dict(inputs)


def prepare_qwenvl_inputs(image: Image.Image, processor: AutoProcessor) -> dict:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": "Describe this image."},
            ],
        }
    ]
    # apply_chat_template builds the prompt string
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor( text=[text_prompt], images=[image], return_tensors="pt")
    return dict(inputs)




# ------------------------------------------------------

def _decode(tokens: torch.Tensor, processor: AutoProcessor) -> list[str]:
    return processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)




# output generation
def generate_blip(model, processor: AutoProcessor, image: Image.Image, decode_cfg: DecodeConfig = DecodeConfig() ) -> dict:
    inputs = prepare_blip_inputs(image, processor)
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    tokens = model.generate.__wrapped__(model,**inputs, **decode_cfg.to_generate_kwargs())
    captions = _decode(tokens.sequences, processor)

    return {"tokens": tokens.sequences, "captions": captions, 'scores' : tokens.scores}


def generate_paligemma(model, processor: AutoProcessor, image: Image.Image, decode_cfg: DecodeConfig = DecodeConfig() ) -> dict:
    inputs = prepare_paligemma_inputs(image, processor)
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        tokens = model.generate(**inputs, **decode_cfg.to_generate_kwargs())

    new_tokens = tokens[:, input_len:]
    captions = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return {"tokens": new_tokens, "captions": captions}


def generate_qwenvl(model, processor: AutoProcessor, image: Image.Image, decode_cfg: DecodeConfig = DecodeConfig() ) -> dict:
    inputs = prepare_qwenvl_inputs(image, processor)
    input_len = inputs["input_ids"].shape[-1]
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        tokens = model.generate(**inputs, **decode_cfg.to_generate_kwargs())

    new_tokens = tokens[:, input_len:]
    captions = processor.batch_decode(new_tokens, skip_special_tokens=True)
    return {"tokens": new_tokens, "captions": captions}



# ---------------------------------------------------------

_GENERATE_FN = {
    "blip":       generate_blip,
    "paligemma":  generate_paligemma,
    "qwen3-vl":   generate_qwenvl,
}


def generate_caption( model, processor: AutoProcessor, image: Image.Image, decode_cfg: DecodeConfig = DecodeConfig(), model_name: str | None = None) -> dict:
    name = (model_name or config.MODEL).lower()

    if name not in _GENERATE_FN:
        raise ValueError(
            f"Unsupported model '{name}'. Choose from: {list(_GENERATE_FN)}"
        )

    fn = _GENERATE_FN[name]
    model.eval()
    return fn(model, processor, image, decode_cfg)
