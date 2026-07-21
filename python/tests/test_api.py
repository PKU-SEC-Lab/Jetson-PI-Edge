import unittest

import numpy as np

import jetson_pi


class TestAPI(unittest.TestCase):
    def test_module_surface(self):
        self.assertTrue(jetson_pi.__doc__)
        self.assertTrue(hasattr(jetson_pi, "PIModel"))
        self.assertTrue(callable(jetson_pi.load_model))

    def test_missing_model_is_reported(self):
        with self.assertRaisesRegex(RuntimeError, "jetson_pi.load_model"):
            jetson_pi.load_model(
                model_path="/path/that/does/not/exist/model.gguf",
                mmproj_path="/path/that/does/not/exist/mmproj.gguf",
                backend="cpu",
                n_views=2,
                image_height=224,
                image_width=224,
            )

    def test_numpy_dependency_available(self):
        images = np.zeros((2, 224, 224, 3), dtype=np.uint8)
        self.assertTrue(images.flags.c_contiguous)


if __name__ == "__main__":
    unittest.main()
