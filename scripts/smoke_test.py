"""End-to-end check with the real models: transcribe a few LibriSpeech clips,
translate them to French, and fail if transcription quality is clearly broken.

Run from the repo root: python scripts/smoke_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jiwer  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

from src import audio as audio_utils  # noqa: E402
from src.pipeline import speech_to_french  # noqa: E402

N_CLIPS = 3
MAX_WER = 0.25

clips = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
clips = clips.cast_column("audio", Audio(decode=False))  # decode with soundfile instead

references, hypotheses = [], []
for clip in clips.select(range(N_CLIPS)):
    audio = audio_utils.decode(clip["audio"]["bytes"] or open(clip["audio"]["path"], "rb").read())
    result = speech_to_french(audio, audio_utils.TARGET_RATE)
    references.append(clip["text"].lower())
    hypotheses.append(result["transcription"])
    print(f"Reference:     {clip['text'].lower()}")
    print(f"Transcription: {result['transcription']}")
    print(f"French:        {result['translation']}\n")
    assert result["translation"], "Translation came back empty."

wer = jiwer.wer(references, hypotheses)
print(f"WER over {N_CLIPS} clips: {wer:.3f}")
assert wer < MAX_WER, f"WER {wer:.3f} is above {MAX_WER}; the pipeline looks broken."
