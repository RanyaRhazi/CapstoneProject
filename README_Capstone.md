# 🗣️ English Speech to French Text Translator

This capstone project implements a multimodal NLP pipeline that transcribes **spoken English** and translates it into **written French** using two open-source pretrained models from Hugging Face.

Built entirely in Python, the system combines real-time speech recognition with neural machine translation, making it a compact proof of concept for multilingual voice interfaces.

---

## ✨ Features

- 🎙️ **Real-time speech transcription** using [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base-960h)
- 🌍 **Neural machine translation** with [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-fr)
- ⚙️ Modular code with clean separation between ASR and translation logic
- 📦 Built using Hugging Face Transformers, PyTorch, and Torchaudio

---

## 🧠 Model Architecture

| Task                  | Model Used                                |
|-----------------------|--------------------------------------------|
| Speech-to-Text (ASR)  | `facebook/wav2vec2-base-960h`              |
| Text Translation      | `Helsinki-NLP/opus-mt-en-fr`               |

---

## 📁 Project Structure

```
CapstoneProject/
├── main.py                        # Entry point – runs ASR + Translation
├── speech_recognition_model.py   # Real-time audio transcription logic
├── machine_translation.py        # Text-to-text translation logic
```

---

## 🚀 How to Run

### ✅ Requirements

- Python 3.8+
- torch
- torchaudio
- sounddevice
- transformers

Install dependencies:

```bash
pip install torch torchaudio sounddevice transformers
```

### ▶️ Run the Translator

```bash
python main.py
```

You’ll be prompted to speak into your microphone for 10 seconds. The system will:
1. Transcribe your English speech
2. Translate the transcription into French
3. Print both results to the console

---

## 📷 Example Output

```
Recording...
Transcribed Speech:
hello my name is ranya and i love working with artificial intelligence
Translation:
bonjour mon nom est ranya et j'aime travailler avec l'intelligence artificielle
```

---

## 📝 Notes

- The models are loaded at runtime — ensure you have an internet connection on the first run.
- The translation supports other language pairs by changing `source_language` and `target_language` codes (e.g., `en-de`, `en-es`).
- You can modify the `duration` variable in `speech_recognition_model.py` to change how long the system listens.

---

## 🧑‍💻 Author

**Ranya Rhazi**  
Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/ranya-rhazi/) | [GitHub](https://github.com/RanyaRhazi)

---

## 📄 License

This project is for academic demonstration purposes only.
