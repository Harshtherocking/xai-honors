from torch.cuda import is_available
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense"
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
    remove_unused_columns=False,  # important when returning dicts
    push_to_hub=False,
)
