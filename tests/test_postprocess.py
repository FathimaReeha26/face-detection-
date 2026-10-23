import unittest
import numpy as np

from detection.postprocess import (
    clip_box,
    corners_to_box,
    intersection_over_union,
    landmark_row,
)


class TestPostprocess(unittest.TestCase):
    def test_clip_box_inside(self):
        box = (10, 20, 30, 40)
        clipped = clip_box(box, width=100, height=100)
        self.assertEqual(clipped, (10, 20, 30, 40))

    def test_clip_box_negative_coords(self):
        # Starts 10px off left edge, spans 50px total -> should start at 0 and have width 40
        box = (-10, 0, 50, 50)
        clipped = clip_box(box, width=100, height=100)
        self.assertEqual(clipped, (0, 0, 40, 50))

        # Starts 15px above top edge, spans 60px height -> should start at 0 and have height 45
        box = (10, -15, 40, 60)
        clipped = clip_box(box, width=100, height=100)
        self.assertEqual(clipped, (10, 0, 40, 45))

    def test_clip_box_overflow(self):
        # Box extends beyond width and height
        box = (80, 80, 50, 50)
        clipped = clip_box(box, width=100, height=100)
        self.assertEqual(clipped, (80, 80, 20, 20))

    def test_clip_box_completely_outside(self):
        # Completely off-screen
        self.assertIsNone(clip_box((-50, -50, 20, 20), width=100, height=100))
        self.assertIsNone(clip_box((120, 120, 20, 20), width=100, height=100))

    def test_clip_box_zero_size(self):
        self.assertIsNone(clip_box((10, 10, 0, 10), width=100, height=100))
        self.assertIsNone(clip_box((10, 10, 10, 0), width=100, height=100))

    def test_corners_to_box(self):
        box = corners_to_box(10.2, 20.4, 60.1, 80.9)
        self.assertEqual(box, (10, 20, 49, 60))

    def test_iou_identical(self):
        box = (10, 10, 20, 20)
        self.assertAlmostEqual(intersection_over_union(box, box), 1.0)

    def test_iou_no_overlap(self):
        box1 = (0, 0, 10, 10)
        box2 = (20, 20, 10, 10)
        self.assertEqual(intersection_over_union(box1, box2), 0.0)

    def test_iou_partial_overlap(self):
        box1 = (0, 0, 10, 10)  # Area 100
        box2 = (5, 0, 10, 10)  # Area 100, Overlap 5x10 = 50. Union = 100 + 100 - 50 = 150
        self.assertAlmostEqual(intersection_over_union(box1, box2), 50.0 / 150.0)

    def test_iou_zero_area(self):
        box1 = (0, 0, 0, 0)
        box2 = (0, 0, 10, 10)
        self.assertEqual(intersection_over_union(box1, box2), 0.0)


if __name__ == "__main__":
    unittest.main()
