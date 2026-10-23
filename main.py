"""Record English speech from the microphone and print its French translation."""
import argparse

from src import speech_recognition_model
from src.pipeline import speech_to_french


def main():
    parser = argparse.ArgumentParser(description="English speech to French text.")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds to record (default: 10)")
    args = parser.parse_args()

    audio = speech_recognition_model.record(args.duration)
    result = speech_to_french(audio)

    print("Transcribed speech:")
    print(result["transcription"] or "(no speech detected)")
    print("French translation:")
    print(result["translation"])


if __name__ == "__main__":
    main()
