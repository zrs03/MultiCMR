import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from utils.augmentation import spatial_affine, random_spatial_affine
from utils.sampling import PatientWindowSampler


class SamplingAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [SimpleNamespace(case_id=patient) for patient, n in [('a', 15), ('b', 9), ('c', 2)] for _ in range(n)]

    def test_patient_quota_unique_and_epoch_reproducible(self):
        sampler = PatientWindowSampler(self.cases, 4, seed=12)
        sampler.set_epoch(3)
        a = list(sampler)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(a), len(set(a)))
        self.assertEqual(Counter(self.cases[i].case_id for i in a), {'a': 4, 'b': 4, 'c': 2})
        self.assertEqual(a, list(sampler))
        sampler.set_epoch(4)
        self.assertNotEqual(a, list(sampler))
        resumed = PatientWindowSampler(self.cases, 4, seed=12)
        resumed.set_epoch(4)
        self.assertEqual(list(sampler), list(resumed))
        self.assertEqual(set(PatientWindowSampler(self.cases, 0)), set(range(26)))

    def test_ddp_shared_selection_and_minimal_padding(self):
        serial = PatientWindowSampler(self.cases, 4, seed=17)
        serial.set_epoch(9)
        expected = list(serial)
        distributed = []
        for rank in range(3):
            sampler = PatientWindowSampler(self.cases, 4, seed=17, rank=rank, num_replicas=3)
            sampler.set_epoch(9)
            distributed.append(list(sampler))
        self.assertEqual([len(x) for x in distributed], [4, 4, 4])
        interleaved = [distributed[r][i] for i in range(4) for r in range(3)]
        self.assertEqual(interleaved[:10], expected)
        self.assertEqual(interleaved[10:], expected[:2])

    def test_affine_translation_matches_center_label_and_all_slices(self):
        label = np.zeros((12, 20), dtype=np.int64)
        label[3:8, 5:12] = 2
        image = np.repeat(label[None], 5, axis=0).astype(np.float32)
        moved, target = spatial_affine(image, label, 0, 1/20, 2/12, 1)
        expected = np.zeros_like(label)
        expected[5:10, 6:13] = 2
        np.testing.assert_array_equal(target, expected)
        np.testing.assert_allclose(moved, np.repeat(expected[None], 5, axis=0), atol=1e-5)
        volume_label = np.repeat(label[None], 5, axis=0)
        sax, sax_target = spatial_affine(image, volume_label, 0, 1/20, 2/12, 1)
        np.testing.assert_allclose(sax, moved)
        np.testing.assert_array_equal(sax_target, np.repeat(expected[None], 5, axis=0))

    def test_rotation_scale_preserve_frames_and_discrete_labels(self):
        label = np.zeros((12, 20), dtype=np.int64)
        label[2:9, 4:15] = 3
        image = np.repeat(label[None], 5, axis=0).astype(np.float32)
        moved, target = spatial_affine(image, label, 9, -.03, .04, .94)
        self.assertEqual(moved.shape, image.shape)
        self.assertEqual(target.shape, label.shape)
        self.assertTrue(set(np.unique(target)) <= {0, 3})
        for frame in moved:
            np.testing.assert_array_equal(frame, moved[0])
        same, same_label = random_spatial_affine(image, label, probability=0)
        np.testing.assert_array_equal(same, image)
        np.testing.assert_array_equal(same_label, label)


if __name__ == '__main__':
    unittest.main()
