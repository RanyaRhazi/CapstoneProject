import unittest

import numpy as np

from src import audio


class AudioUtilsTest(unittest.TestCase):
    def test_resamples_to_16k_and_mono(self):
        stereo_44k = np.zeros((44100, 2), dtype=np.float32)
        out = audio.to_mono_16k(stereo_44k, 44100)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.size, 16000)

    def test_noise_hits_target_snr(self):
        rng = np.random.default_rng(1)
        clean = rng.normal(0, 0.1, 160000).astype(np.float32)
        noisy = audio.add_noise(clean, snr_db=10)
        snr = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
        self.assertAlmostEqual(snr, 10, delta=0.2)

    def test_speed_change_length(self):
        clip = np.zeros(16000, dtype=np.float32)
        self.assertEqual(audio.change_speed(clip, 1.25).size, 12800)
        self.assertEqual(audio.change_speed(clip, 0.8).size, 20000)


if __name__ == "__main__":
    unittest.main()
