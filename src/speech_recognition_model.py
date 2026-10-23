"""English speech recognition with Wav2Vec2 (facebook/wav2vec2-base-960h).

The model is loaded once, on first use, so importing this module is cheap.
"""
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000  # the model was trained on 16 kHz audio

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        _processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def record(duration: float = 10.0):
    """Record mono audio from the default microphone and return it as a 1-D float array."""
    import sounddevice as sd

    print(f"Recording for {duration:g} seconds...")
    audio = sd.rec(int(SAMPLE_RATE * duration), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def transcribe(audio, sample_rate: int = SAMPLE_RATE) -> str:
    """Transcribe a 1-D array of 16 kHz English speech to lowercase text."""
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz.")

    import torch

    processor, model = _load_model()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    # The model outputs uppercase letters; lowercase reads better and suits the translator.
    return processor.batch_decode(predicted_ids)[0].strip().lower()
