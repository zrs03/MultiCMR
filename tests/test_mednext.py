"""MedNeXt-M scale integration and reversible spatial preprocessing."""
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from models import MMFormerAnatomy
from preprocess.processing import prepare_nifti, restore_native_labels


class MedNeXtTests(unittest.TestCase):
    def test_all_m_stages_and_full_resolution_token_count(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2).eval()
        encoder = model.encoders['sa']
        self.assertEqual([len(s) for s in encoder.backbone.stages], [3,4,4,4,4])
        self.assertEqual([len(s) for s in model.decoder.stages], [4,4,4,3])
        with torch.no_grad():
            features, tokens, shape = encoder(torch.randn(1,1,5,256,256))
        self.assertEqual([tuple(f.shape[1:]) for f in features],
                         [(2,256,256),(4,128,128),(8,64,64),(16,32,32),(32,16,16)])
        self.assertEqual(tuple(tokens.shape), (1,256,8))
        self.assertEqual(tuple(shape), (16,16))
        self.assertEqual(model.context_projection['sa'].out_channels, 32)

    def test_padding_and_native_geometry_roundtrip(self):
        # Odd difference exercises asymmetric center padding and crop offsets.
        raw = np.zeros((5,8,3), dtype=np.float32)
        raw[1:4,2:6,:] = 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'image.nii.gz'
            affine = np.diag([1.2,1.3,8.,1.])
            nib.save(nib.Nifti1Image(raw, affine), path)
            labels, saved_affine, original = prepare_nifti(path, 'sa', (5,256,256), True)
            restored = restore_native_labels(labels, original).numpy()
            np.testing.assert_array_equal(np.moveaxis(restored,0,-1), raw)
            np.testing.assert_allclose(saved_affine, affine)
            self.assertTrue((labels[:, :32] == 0).all())
            self.assertTrue((labels[:, -64:] == 0).all())


if __name__ == '__main__':
    unittest.main()
