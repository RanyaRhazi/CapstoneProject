import sounddevice as sd
import numpy as np
import speech_recognition as sr
#import librosa
import noisereduce as nr


def transcribe_realtime_speech():
    # Audio sampling parameters
    sample_rate = 44100  
    duration = 10
    
    # Audio Recording
    print("Recording...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1)
    sd.wait()

    audio_data = audio_data[:, 0]
    
    noisy_part = audio_data
    reduced_noise = nr.reduce_noise(y=noisy_part, sr=sample_rate)
    
    # Speech recognizer
    recognizer = sr.Recognizer()

    # Audio Data COnversion
    audio_source = sr.AudioData(
        reduced_noise.tobytes(),
        sample_rate=sample_rate,
        sample_width=reduced_noise.dtype.itemsize  # Sample Width
    )
    

    # Speech recognition
    try:
        transcribed_text = recognizer.recognize_google(audio_source, language="en-US")
        print("Transcribed Speech:")
        print(transcribed_text)
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    transcribe_realtime_speech()
