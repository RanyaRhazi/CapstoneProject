from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, source_language, target_language):
    # Load the translation model and tokenizer
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Tokenize the input text and generate translation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translation = model.generate(**inputs)

    # Decode the translation output
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    return translated_text
