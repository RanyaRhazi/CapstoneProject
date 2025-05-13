import torchaudio
import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load the Wav2Vec2 model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

def transcribe_realtime_speech():
    # Audio sampling parameters
    sample_rate = 16000
    duration = 10

    # Audio Recording
    print("Recording...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()

    audio_data = audio_data[:, 0]

    # Convert the audio data to a format recognized by the model
    audio_signal = torch.tensor(audio_data)

    # Perform real-time speech recognition
    with torch.no_grad():
        input_values = tokenizer(audio_data, return_tensors="pt").input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)

    return transcription[0]
