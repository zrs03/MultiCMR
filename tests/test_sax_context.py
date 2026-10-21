"""SAX neighborhood semantics, phase isolation and center supervision."""
import unittest
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from models import MMFormerAnatomy
from preprocess.processing import sax_context_indices, validate_sax_geometry, prepare_nifti, validate_output_sizes
from utils.inference import predict_cine
from test import evaluate_prediction


class SAXContextTests(unittest.TestCase):
    def test_spatial_and_temporal_neighbors_are_distinct(self):
        # Each frame identifies z and t: intensity = 100*z+t.
        volume = np.array([100*z+t for z in range(3) for t in range(4)])
        np.testing.assert_array_equal(volume[sax_context_indices(2, 1, 4, 3)], [102,101,103,2,202])
        # Temporal boundary must not spill into another spatial slice.
        np.testing.assert_array_equal(volume[sax_context_indices(0, 1, 4, 3)], [100,101,101,0,200])
        # Spatial reflection; do not wrap from heart base to apex.
        np.testing.assert_array_equal(volume[sax_context_indices(3, 0, 4, 3)], [3,2,2,103,103])
        np.testing.assert_array_equal(sax_context_indices(0, 0, 1, 1), [0]*5)

    def test_no_phase_axis_interpolation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'sax.nii.gz'
            array = np.broadcast_to(np.arange(12, dtype=np.float32), (7,9,12)).copy()
            nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            labels, _, original = prepare_nifti(path, 'sa', (5,16,24), is_label=True)
            self.assertEqual(labels.shape, (12,16,24))
            self.assertEqual(original, (7,9,12))
            np.testing.assert_array_equal(labels[:,0,0], np.arange(12))
        for phases in (None, 0, -1, 2.5, True):
            with self.assertRaises(ValueError): validate_sax_geometry(12, phases)
        with self.assertRaises(ValueError): validate_sax_geometry(13, 4)
        with self.assertRaises(ValueError):
            validate_output_sizes({'2ch': (5,16,16), '4ch': (5,16,16), 'sa': (128,16,16)})

    def test_partial_slice_truncation_and_background_restore(self):
        self.assertEqual(validate_sax_geometry(290, 25, allow_partial=True), 11)
        with self.assertRaises(ValueError):
            validate_sax_geometry(3, 4, allow_partial=True)

        class ForegroundModel(torch.nn.Module):
            temporal_size = 5
            def __init__(self):
                super().__init__()
                self.seen = []
            def forward(self, images, mask):
                self.seen.extend(images['sa'][0, 0, :, 0, 0].tolist())
                logits = torch.zeros(1, 4, 2, 2)
                logits[:, 1] = 10
                return {'logits': {v: logits for v in images}}

        model = ForegroundModel().eval()
        images = {v: torch.zeros(1, 1, 4, 2, 2) for v in ('2ch', '4ch')}
        images['sa'] = torch.arange(10).float().view(1, 1, 10, 1, 1).expand(-1, -1, -1, 2, 2)
        pred = predict_cine(model, images, torch.tensor([[0, 0, 1]]), sax_phases=4)['sa']
        self.assertEqual(pred.shape, (1, 10, 2, 2))
        self.assertTrue((pred[:, :8] == 1).all())
        self.assertTrue((pred[:, 8:] == 0).all())
        self.assertEqual(len(model.seen), 8 * 5)
        self.assertLess(max(model.seen), 8)
        from unittest.mock import patch
        observed = []
        def metric(p, y, classes):
            observed.extend(y[:, 0, 0].tolist())
            return (1., 1., 0.)
        target = torch.arange(10).view(10, 1, 1).expand(-1, 2, 2)
        with patch('test.multiclass_dice_iou_hd95', side_effect=metric):
            evaluate_prediction(pred[0], target, 'sa', 4, sax_phases=4)
        self.assertEqual(sorted(observed), list(range(10)))

    def test_sax_center_head_uses_all_five_channels(self):
        torch.manual_seed(16)
        model = MMFormerAnatomy(base=2, dim=8, heads=2).eval()
        x = {v: torch.randn(1,1,5,16,16, requires_grad=True) for v in ('2ch','4ch','sa')}
        result = model(x, torch.tensor([[0,0,1]]))
        self.assertEqual(result['logits']['sa'].shape, (1,4,16,16))
        self.assertEqual(result['patch_shapes']['sa'], (2,2))
        result['logits']['sa'].square().mean().backward()
        for channel in range(5):
            self.assertGreater(x['sa'].grad[:,:,channel].abs().sum().item(), 0)
        self.assertIsNone(x['2ch'].grad)

    def test_sax_metrics_evaluate_spatial_volume_per_phase(self):
        from unittest.mock import patch
        pred = torch.arange(12).reshape(12,1,1)
        observed = []
        def metric(p, y, classes):
            observed.append(p[:,0,0].tolist())
            return (1.,1.,0.)
        with patch('test.multiclass_dice_iou_hd95', side_effect=metric):
            self.assertEqual(evaluate_prediction(pred,pred,'sa',4,sax_phases=4), (1.,1.,0.))
        self.assertEqual(observed, [[0,4,8],[1,5,9],[2,6,10],[3,7,11]])

    def test_inference_contexts_match_training_and_require_metadata(self):
        class Recorder(torch.nn.Module):
            temporal_size = 5
            def __init__(self):
                super().__init__(); self.frames=[]
            def forward(self, images, mask):
                self.frames.append(images['sa'][0,0,:,0,0].tolist())
                return {'logits': {v: torch.zeros(1,4,8,8) for v in images}}
        model=Recorder().eval()
        images={v: torch.zeros(1,1,3,8,8) for v in ('2ch','4ch')}
        images['sa']=torch.tensor([100*z+t for z in range(3) for t in range(4)]).float().view(1,1,12,1,1).expand(-1,-1,-1,8,8)
        with self.assertRaisesRegex(ValueError, 'phase count'):
            predict_cine(model, images, torch.tensor([[0,0,1]]))
        outputs=predict_cine(model, images, torch.tensor([[0,0,1]]), sax_phases=4)
        self.assertEqual(outputs['sa'].shape,(1,12,8,8))
        self.assertEqual(len(model.frames),12)
        self.assertIn([102,101,103,2,202],model.frames)
        self.assertEqual({f[0] for f in model.frames}, {float(100*z+t) for z in range(3) for t in range(4)})


if __name__ == '__main__': unittest.main()
