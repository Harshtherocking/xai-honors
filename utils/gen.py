import torch



def generate_caption_logits (inputs, model, processor, max_length=30):

    pixels = inputs["pixel_values"]
    # generated_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]
    generated_ids = torch.tensor([
        [101]
    ])

    for i in range(max_length):
        # zero all collected grads
        # model.zero_grad()

        outputs = model(pixel_values = pixels, input_ids = generated_ids)

        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

        if next_token.item() == processor.tokenizer.sep_token_id:
            break

    return generated_ids
