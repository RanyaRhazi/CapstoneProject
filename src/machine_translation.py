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

    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    output_ids = model.generate(**inputs)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
