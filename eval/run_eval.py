"""Reproducible evaluation of the cascaded pipeline on public data.

ASR (Wav2Vec2): WER and CER on LibriSpeech test-clean, on clean audio and under
added white noise (20, 10, 5 dB SNR) and speed changes (0.9x, 1.1x).
MT (MarianMT): BLEU and chrF on WMT14 newstest2014 English-to-French.

Run from the repo root:
    python eval/run_eval.py --asr-n 200 --mt-n 500
Writes eval/results.json and prints a Markdown table.
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jiwer  # noqa: E402
import sacrebleu  # noqa: E402
from datasets import Audio, load_dataset  # noqa: E402

from src import audio as audio_utils  # noqa: E402
from src import machine_translation, speech_recognition_model  # noqa: E402

CONDITIONS = [
    ("Clean", lambda a: a),
    ("White noise, 20 dB SNR", lambda a: audio_utils.add_noise(a, 20)),
    ("White noise, 10 dB SNR", lambda a: audio_utils.add_noise(a, 10)),
    ("White noise, 5 dB SNR", lambda a: audio_utils.add_noise(a, 5)),
    ("Speed 0.9x", lambda a: audio_utils.change_speed(a, 0.9)),
    ("Speed 1.1x", lambda a: audio_utils.change_speed(a, 1.1)),
]


def evaluate_asr(n: int) -> dict:
    stream = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
    stream = stream.cast_column("audio", Audio(decode=False))  # decode with soundfile instead
    clips = []
    for row in stream:
        clip = audio_utils.decode(row["audio"]["bytes"] or open(row["audio"]["path"], "rb").read())
        clips.append((clip, row["text"].lower()))
        if len(clips) == n:
            break

    references = [text for _, text in clips]
    results = []
    for name, perturb in CONDITIONS:
        start = time.perf_counter()
        hypotheses = [speech_recognition_model.transcribe(perturb(clip)) for clip, _ in clips]
        seconds = time.perf_counter() - start
        results.append({
            "condition": name,
            "wer": jiwer.wer(references, hypotheses),
            "cer": jiwer.cer(references, hypotheses),
            "seconds": seconds,
        })
        print(f"ASR {name}: WER {results[-1]['wer']:.4f}, CER {results[-1]['cer']:.4f}", flush=True)

    audio_seconds = sum(len(clip) for clip, _ in clips) / audio_utils.TARGET_RATE
    return {
        "dataset": "LibriSpeech test-clean (openslr/librispeech_asr), first utterances in dataset order",
        "utterances": len(clips),
        "audio_seconds": audio_seconds,
        "conditions": results,
    }


def evaluate_mt(n: int, batch_size: int = 16) -> dict:
    stream = load_dataset("wmt/wmt14", "fr-en", split="test", streaming=True)
    pairs = []
    for row in stream:
        pairs.append((row["translation"]["en"], row["translation"]["fr"]))
        if len(pairs) == n:
            break

    sources = [en for en, _ in pairs]
    references = [fr for _, fr in pairs]
    start = time.perf_counter()
    hypotheses = []
    for i in range(0, len(sources), batch_size):
        hypotheses.extend(machine_translation.translate_batch(sources[i:i + batch_size]))
    seconds = time.perf_counter() - start

    return {
        "dataset": "WMT14 newstest2014 en-fr (wmt/wmt14), first sentences in dataset order",
        "sentences": len(pairs),
        "bleu": sacrebleu.corpus_bleu(hypotheses, [references]).score,
        "chrf": sacrebleu.corpus_chrf(hypotheses, [references]).score,
        "seconds": seconds,
    }


def to_markdown(results: dict) -> str:
    asr, mt = results["asr"], results["mt"]
    lines = [
        f"ASR: {asr['utterances']} utterances ({asr['audio_seconds'] / 60:.1f} min) of {asr['dataset']}",
        "",
        "| Condition | WER | CER |",
        "|---|---|---|",
    ]
    lines += [f"| {c['condition']} | {c['wer']:.1%} | {c['cer']:.1%} |" for c in asr["conditions"]]
    lines += [
        "",
        f"MT: {mt['sentences']} sentences of {mt['dataset']}",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| BLEU | {mt['bleu']:.1f} |",
        f"| chrF | {mt['chrf']:.1f} |",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-n", type=int, default=200)
    parser.add_argument("--mt-n", type=int, default=500)
    args = parser.parse_args()

    results = {
        "asr_model": speech_recognition_model.MODEL_NAME,
        "mt_model": machine_translation.MODEL_NAME,
        "python": platform.python_version(),
        "asr": evaluate_asr(args.asr_n),
        "mt": evaluate_mt(args.mt_n),
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results))
    print(to_markdown(results))


if __name__ == "__main__":
    main()
