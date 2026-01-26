import evaluate

BLEU = evaluate.load("bleu")
ROUGE = evaluate.load("rouge")
CHRF = evaluate.load("chrf")


def bleu_score (predictions, references) -> float:
    assert len(predictions) == len(references) , "predictions and references must have same length"
    return BLEU.compute(predictions = predictions, references = references)["bleu"]

def rouge_score (predictions, references) -> float:
    assert len(predictions) == len(references) , "predictions and references must have same length"
    return ROUGE.compute(predictions = predictions, references = references)["rougeLsum"]

def chrf_score (predictions, references) -> float:
    assert len(predictions) == len(references) , "predictions and references must have same length"
    return CHRF.compute(predictions = predictions, references = references, lowercase = True, word_order = 2)["score"] / 100


if __name__ == "__main__":
    prediction = ["hi", "hello"]
    reference = [["hello", "hi"] , ["hi", "hello"]]

    bleu = bleu_score(prediction, reference)
    rouge = rouge_score(prediction, reference)
    chrf = chrf_score(prediction, reference)

    print(bleu, rouge, chrf)
