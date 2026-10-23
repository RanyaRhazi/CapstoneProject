"""English-to-French translation with MarianMT (Helsinki-NLP/opus-mt-en-fr).

The model is loaded once, on first use, so importing this module is cheap.
"""
MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import MarianMTModel, MarianTokenizer

        _tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        _model = MarianMTModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def translate(text: str) -> str:
    """Translate English text to French."""
    if not text.strip():
        return ""
    return translate_batch([text])[0]


def translate_batch(texts: list[str]) -> list[str]:
    """Translate a list of English sentences to French in one forward pass."""
    if not texts:
        return []

    tokenizer, model = _load_model()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    output_ids = model.generate(**inputs)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)
