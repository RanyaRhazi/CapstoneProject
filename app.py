"""HTTP API for the English speech to French text pipeline.

Run: python app.py   (serves on http://localhost:5000)
"""
import os

from flask import Flask, jsonify, request

from src import audio as audio_utils
from src.pipeline import speech_to_french

MAX_UPLOAD_MB = 20

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/translate")
def translate():
    upload = request.files.get("audio")
    if upload is None:
        return jsonify({"error": "Send the recording as multipart form data in a field named 'audio'."}), 400

    try:
        audio = audio_utils.decode(upload.read())
    except Exception:
        return jsonify({"error": "Could not decode the audio. Send a WAV, FLAC or OGG file."}), 400

    if audio.size == 0:
        return jsonify({"error": "The audio file is empty."}), 400

    return jsonify(speech_to_french(audio, audio_utils.TARGET_RATE))


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large; the limit is {MAX_UPLOAD_MB} MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
