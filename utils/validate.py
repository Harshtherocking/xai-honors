import torch
from datasets import Dataset

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration, AutoProcessor
)


from utils.evaluation import bleu_score, rouge_score, chrf_score

def validate (output : torch.Tensor, val_dataset : Dataset, processor : AutoProcessor) -> tuple[float, float, float]:
    gen_tokens = output["logits"].argmax(dim=-1)
    gen_caption = processor.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    gt_caption = processor.tokenizer.batch_decode(list(val_dataset["input_ids"]), skip_special_tokens=True)

    # even for single ref sentence use list of list ---- documentation
    gt_caption = [[_] for _ in gt_caption]

    bleu = bleu_score(gen_caption, gt_caption)
    rogue = rouge_score(gen_caption, gt_caption)
    chrf = chrf_score(gen_caption, gt_caption)
    return bleu, rogue, chrf


