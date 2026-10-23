"""API tests with the models and audio decoding mocked."""
import io
import unittest
from unittest import mock

import numpy as np

import app as api


class TranslateEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()

    def post_audio(self, data=b"fake-audio"):
        return self.client.post(
            "/translate",
            data={"audio": (io.BytesIO(data), "clip.wav")},
            content_type="multipart/form-data",
        )

    def test_health(self):
        self.assertEqual(self.client.get("/health").get_json(), {"status": "ok"})

    def test_returns_transcription_and_translation(self):
        result = {"transcription": "hello", "translation": "bonjour"}
        with mock.patch.object(api.audio_utils, "decode", return_value=np.zeros(16000, dtype=np.float32)), \
             mock.patch.object(api, "speech_to_french", return_value=result) as pipeline:
            response = self.post_audio()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), result)
        pipeline.assert_called_once()

    def test_missing_file(self):
        response = self.client.post("/translate", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)

    def test_undecodable_file(self):
        with mock.patch.object(api.audio_utils, "decode", side_effect=RuntimeError("bad file")):
            response = self.post_audio()
        self.assertEqual(response.status_code, 400)

    def test_empty_audio(self):
        with mock.patch.object(api.audio_utils, "decode", return_value=np.zeros(0, dtype=np.float32)):
            response = self.post_audio()
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
