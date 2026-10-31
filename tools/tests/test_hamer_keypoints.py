import unittest

import hamer_keypoints  # 사용자 설치 numpy 경로도 여기서 붙는다(HaMeR 자체는 load에서만 불러온다)

import numpy as np  # noqa: E402


class HandBoxTest(unittest.TestCase):
    def test_box_is_the_pixel_extent_of_the_landmarks(self):
        pts = np.array([[0.25, 0.5], [0.5, 0.25], [0.75, 0.75]] + [[0.5, 0.5]] * 18)
        np.testing.assert_allclose(hamer_keypoints.hand_box(pts, 640, 480), [[160.0, 120.0, 480.0, 360.0]])
        self.assertEqual((1, 4), hamer_keypoints.hand_box(pts, 640, 480).shape)
        self.assertEqual(np.float32, hamer_keypoints.hand_box(pts, 640, 480).dtype)


if __name__ == "__main__":
    unittest.main()
