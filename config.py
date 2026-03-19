from torch.cuda import is_available
from peft import LoraConfig


HF_TOKEN = "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"

MODEL = "blip"
LOAD_PRETRAINED_MODEL = True


IMAGE_SIZE = 224

DEVICE = "cuda" if is_available() else "cpu"
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 100
MAX_TARGET_LENGTH = 64
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

LOG_DIR = f"./{MODEL}-logs"
OUTPUT_DIR = f"./{MODEL}-ft"

TARGET_MODULES = {
    "blip" : (
        r"^text_decoder\.bert\.encoder\.layer\..*\."
        r"(query|key|value|dense)$"
        r"|^text_decoder\.cls\.predictions\.(transform\.dense|decoder)$"
    ),

    "paligemma" : (
        r"^model\.language_model\..*(q_proj|k_proj|v_proj|o_proj|"
        r"gate_proj|up_proj|down_proj)$"
        r"|^lm_head$"
    ),

    "qwen3-vl" : (
        r"^model\.language_model\.layers\..*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        r"|^lm_head$"
    )
}


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES[MODEL],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    # task_type=TaskType.CAUSAL_LM,  # for encoder-decoder captioning
)


# prior
ALPHA = 0.8