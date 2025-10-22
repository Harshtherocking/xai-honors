from torch.cuda import is_available
from transformers import TrainingArguments
from peft import LoraConfig

DEVICE = "cuda" if is_available() else "cpu"
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 3
MAX_TARGET_LENGTH = 32
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

OUTPUT_DIR = "./output"

TARGET_MODULES = {
    "blip" : [
        "encoder.layer[:].attention.self.query",
        "encoder.layer[:].attention.self.key",
        "encoder.layer[:].attention.self.value",
        "encoder.layer[:].attention.output.dense"
    ],
    "paligemma" : [
        "language_model.layers[:].q_proj",
        "language_model.layers[:].k_proj",
        "language_model.layers[:].v_proj"
        "language_model.layers[:].mlp"
        "lm_head"
    ],
    "qwen3-vl" : [
        "language_model.layers[:].q_proj"
        "language_model.layers[:].k_proj"
        "language_model.layers[:].v_proj"
        "language_model.layers[:].o_proj"
        "language_model.layers[:].mlp"
        "language_model.layers[:].lm_head"
    ]
}


lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES["blip"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SEQ_2_SEQ_LM",  # for encoder-decoder captioning
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    # evaluation_strategy="no",
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    fp16=is_available(),
    save_total_limit=2,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    remove_unused_columns=True,  # important when returning dicts
    push_to_hub=False,
)
