from src import speech_recognition_model
from src import machine_translation

if __name__ == "__main__":
    try:
        transcribed_speech = speech_recognition_model.transcribe_realtime_speech()
        source_language = "en"  # Change to the actual source language
        target_language = "fr"  # Change to the desired target language

        translated_speech = machine_translation.translate_text(transcribed_speech, source_language, target_language)

        print("Transcribed Speech:")
        print(transcribed_speech)
        print("Translation:")
        print(translated_speech)
    except Exception as e:
        print("An error occurred:", e)
