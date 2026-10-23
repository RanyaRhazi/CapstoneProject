"""Cascaded pipeline: English speech -> English text (Wav2Vec2) -> French text (MarianMT)."""
from src import machine_translation, speech_recognition_model


def speech_to_french(audio, sample_rate: int = speech_recognition_model.SAMPLE_RATE) -> dict:
    """Return the English transcription and its French translation for one audio clip."""
    transcription = speech_recognition_model.transcribe(audio, sample_rate)
    translation = machine_translation.translate(transcription) if transcription else ""
    return {"transcription": transcription, "translation": translation}
