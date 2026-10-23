"""Unit tests for the pipeline wiring. Models are mocked, so these run without
downloading anything. Run with: python -m unittest discover -s tests -v
"""
import unittest
from unittest import mock

from src import machine_translation, pipeline, speech_recognition_model


class ImportTest(unittest.TestCase):
    def test_import_does_not_load_models(self):
        self.assertIsNone(speech_recognition_model._model)
        self.assertIsNone(machine_translation._model)


class TranscribeTest(unittest.TestCase):
    def test_rejects_wrong_sample_rate(self):
        with self.assertRaises(ValueError):
            speech_recognition_model.transcribe([0.0] * 100, sample_rate=44100)


class TranslateTest(unittest.TestCase):
    def test_empty_text_skips_model(self):
        with mock.patch.object(machine_translation, "_load_model") as load:
            self.assertEqual(machine_translation.translate("   "), "")
            load.assert_not_called()


class PipelineTest(unittest.TestCase):
    def test_transcribes_then_translates(self):
        with mock.patch.object(speech_recognition_model, "transcribe", return_value="hello world") as asr, \
             mock.patch.object(machine_translation, "translate", return_value="bonjour le monde") as mt:
            result = pipeline.speech_to_french([0.0] * 16000)

        asr.assert_called_once()
        mt.assert_called_once_with("hello world")
        self.assertEqual(result, {"transcription": "hello world", "translation": "bonjour le monde"})

    def test_no_speech_skips_translation(self):
        with mock.patch.object(speech_recognition_model, "transcribe", return_value=""), \
             mock.patch.object(machine_translation, "translate") as mt:
            result = pipeline.speech_to_french([0.0] * 16000)

        mt.assert_not_called()
        self.assertEqual(result, {"transcription": "", "translation": ""})


if __name__ == "__main__":
    unittest.main()
