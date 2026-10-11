"""Regression tests for time-axis handling, heterogeneous patches and masking."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import MMFormerAnatomy, masked_segmentation_loss
from models.model import CineViewEncoder, HybridViewEncoder, TransformerBlock
from preprocess.processing import DEFAULT_OUTPUT_SIZES, prepare_nifti, to_model_axes
from train import boundary_loss_per_sample, run_one_epoch
from utils.dataset import CaseItem, MultiSourceNiftiDataset3D
from utils.inference import predict_cine
from utils.metrics import dice_ce_loss_per_sample

VIEWS = ('2ch', '4ch', 'sa')
CLASSES = {'2ch': 3, '4ch': 6, 'sa': 4}


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(3)
        np.random.seed(3)

    def test_raw_time_axis_and_nearest_labels(self):
        raw = np.broadcast_to(np.arange(5, dtype=np.float32), (7, 9, 5)).copy()
        self.assertEqual(to_model_axes(raw, '2ch').shape, (5, 7, 9))
        np.testing.assert_array_equal(to_model_axes(raw, '2ch')[:, 0, 0], np.arange(5))
        np.testing.assert_array_equal(to_model_axes(raw[:, :, None], '4ch'), to_model_axes(raw, '2ch'))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cine.nii.gz'
            nib.save(nib.Nifti1Image(raw, np.eye(4)), path)
            label, _, shape = prepare_nifti(path, '2ch', (10, 7, 9), is_label=True)
            np.testing.assert_array_equal(label[:, 0, 0], np.repeat(np.arange(5), 2))
            self.assertEqual(shape, (7, 9, 5))
        self.assertEqual(DEFAULT_OUTPUT_SIZES['2ch'], (64, 128, 128))
        self.assertEqual(DEFAULT_OUTPUT_SIZES['sa'], (128, 128, 128))

    def make_case(self, directory):
        sizes = {'2ch': (4, 16, 24), '4ch': (4, 16, 24), 'sa': (16, 16, 24)}
        paths = {}
        for v, size in sizes.items():
            label = np.indices(size).sum(axis=0) % CLASSES[v]
            path = Path(directory) / f'{v}.npz'
            np.savez_compressed(path, image=label.astype(np.float32), label=label,
                                affine=np.eye(4), original_shape=np.array((size[1], size[2], size[0])),
                                view=np.array(v), layout=np.array('DHW' if v == 'sa' else 'THW'))
            paths[v] = str(path)
        return CaseItem('001', paths, paths), sizes

    def test_full_sequences_pairing_and_collation(self):
        with tempfile.TemporaryDirectory() as d:
            case, sizes = self.make_case(d)
            full = MultiSourceNiftiDataset3D([case], sizes)[0]
            dataset = MultiSourceNiftiDataset3D([case], sizes, augment=True)
            for _ in range(4):
                item = dataset[0]
                for v in VIEWS:
                    torch.testing.assert_close(item['image'][v][0], item['masks'][v].float())
            batch = next(iter(DataLoader(dataset, batch_size=1)))
            self.assertEqual(batch['image']['2ch'].shape, (1, 1, 4, 16, 24))
            self.assertEqual(batch['image']['sa'].shape, (1, 1, 16, 16, 24))
            # Spatial flips applied to an entire cine never reorder its frames.
            with patch('numpy.random.rand', return_value=0.):
                augmented = MultiSourceNiftiDataset3D([case], sizes, augment=True)[0]
            torch.testing.assert_close(augmented['image']['2ch'], full['image']['2ch'].flip((-2, -1)))

    def test_seven_masks_native_patches_and_gradients(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2, token_grid=(2, 2, 2))
        self.assertIsInstance(model.encoders['2ch'].c1.conv[0], torch.nn.Conv2d)
        self.assertIsInstance(model.encoders['4ch'].patch_embed, torch.nn.Conv2d)
        self.assertIsInstance(model.encoders['sa'].c1.conv[0], torch.nn.Conv3d)
        present = torch.tensor([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]], dtype=torch.bool)
        x = {v: torch.randn(7, 1, *((17, 25, 29) if v == 'sa' else (2 if v == '2ch' else 3, 25, 29))) for v in VIEWS}
        y = {v: torch.randint(CLASSES[v], (7, *x[v].shape[2:])) for v in VIEWS}
        result = model(x, present)
        self.assertEqual(result['patch_shapes'], {'2ch': (2,4,4), '4ch': (3,4,4), 'sa': (3,4,4)})
        for i,v in enumerate(VIEWS):
            self.assertEqual(result['logits'][v].shape, (7, CLASSES[v], *x[v].shape[2:]))
            self.assertFalse(result['logits'][v][~present[:,i]].any())
        loss = masked_segmentation_loss(result, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for p in (model.inter.fusion_grid, model.inter.anatomy_tokens,
                  model.encoders['2ch'].temporal_pos,
                  model.encoders['2ch'].c1.conv[0].weight, model.encoders['sa'].c1.conv[0].weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0.)
        model.eval()
        with torch.no_grad():
            baseline = model(x, present)
            corrupted = {v: a.clone() for v,a in x.items()}
            for i,v in enumerate(VIEWS):
                corrupted[v][~present[:,i]] = float('nan')
            actual = model(corrupted, present)
        for v in VIEWS:
            torch.testing.assert_close(actual['logits'][v], baseline['logits'][v])
        with self.assertRaises(ValueError):
            model(x, torch.zeros_like(present))

    def test_2d_3d_losses_and_training_step(self):
        for spatial in ((16, 24), (12, 16, 24)):
            logits = torch.randn(2, 3, *spatial, requires_grad=True)
            target = torch.randint(3, (2, *spatial))
            loss = dice_ce_loss_per_sample(logits, target, 3) + boundary_loss_per_sample(logits, target, 3)
            self.assertEqual(loss.shape, (2,))
            loss.mean().backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
        with tempfile.TemporaryDirectory() as d:
            case,sizes = self.make_case(d)
            loader = DataLoader(MultiSourceNiftiDataset3D([case], sizes), batch_size=1)
            model = MMFormerAnatomy(base=2, dim=8, heads=2)
            optimizer = torch.optim.Adam(model.parameters())
            with patch('train.sample_modality_mask', return_value=torch.tensor([[1.,0.,1.]])):
                loss,stats = run_one_epoch(model, loader, optimizer, torch.device('cpu'), VIEWS, CLASSES, True)
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(stats['4ch'], {'dice': 0., 'iou': 0.})
            loss,_ = run_one_epoch(model, loader, optimizer, torch.device('cpu'), VIEWS, CLASSES, False)
            self.assertTrue(np.isfinite(loss))

    def test_temporal_concatenation_preserves_every_frame(self):
        encoder = CineViewEncoder(1, 2, 8, (2, 2, 2), 1, 2, temporal_size=3).eval()
        x = torch.randn(2, 1, 3, 16, 24)
        with torch.no_grad():
            encoder.temporal_pos.zero_()
            skips, tokens, shape = encoder(x)
            frame_tokens = [HybridViewEncoder.forward(encoder, x[:, :, t])[1] for t in range(3)]
        self.assertEqual(shape, (3, 2, 3))
        self.assertEqual(tokens.shape, (2, 18, 8))
        self.assertEqual(skips[0].shape, (2, 2, 3, 16, 24))
        torch.testing.assert_close(tokens, torch.cat(frame_tokens, dim=1))

    def test_frames_interact_after_fusion(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2, temporal_size=3).eval()
        x = {v: torch.randn(1, 1, 3 if v != 'sa' else 16, 16, 24) for v in VIEWS}
        present = torch.tensor([[1, 0, 0]])
        with torch.no_grad():
            baseline = model(x, present)['logits']['2ch'][:, :, 0]
            x['2ch'][:, :, 2] = torch.randn_like(x['2ch'][:, :, 2]) * 4
            actual = model(x, present)['logits']['2ch'][:, :, 0]
        self.assertGreater(float((actual - baseline).abs().max()), 1e-6)

    def test_query_chunk_attention_matches_full_attention(self):
        full = TransformerBlock(8, 2, dropout=0., query_chunk_size=100)
        chunked = TransformerBlock(8, 2, dropout=0., query_chunk_size=3)
        chunked.load_state_dict(full.state_dict())
        x = torch.randn(2, 11, 8, requires_grad=True)
        xc = x.detach().clone().requires_grad_()
        mask = torch.zeros(2, 11, dtype=torch.bool)
        mask[0, 4:8] = True
        baseline = full(x, key_padding_mask=mask)
        actual = chunked(xc, key_padding_mask=mask)
        torch.testing.assert_close(actual, baseline)
        baseline.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(xc.grad, x.grad)
        for a,b in zip(chunked.parameters(), full.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)

    def test_boundary_loss_never_uses_temporal_distance(self):
        logits = torch.randn(2, 3, 4, 12, 16, requires_grad=True)
        target = torch.randint(3, (2, 4, 12, 16))
        actual = boundary_loss_per_sample(logits, target, 3, framewise=True)
        expected = torch.stack([boundary_loss_per_sample(logits[:, :, t], target[:, t], 3)
                                for t in range(4)], dim=1).mean(1)
        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_complete_cine_and_sax_only_inference(self):
        class FrameModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x, mask):
                self.calls += 1
                self.assert_shapes(x)
                logits = {v: torch.cat((-a, a), dim=1) for v,a in x.items()}
                return {'logits': logits}
            @staticmethod
            def assert_shapes(x):
                assert x['2ch'].ndim == 5 and x['sa'].ndim == 5
        model = FrameModel().eval()
        images = {v: torch.randn(2, 1, 4, 8, 8) for v in VIEWS}
        present = torch.tensor([[1,1,1],[0,1,1]], dtype=torch.bool)
        result = predict_cine(model, images, present)
        self.assertEqual(model.calls, 1)
        for i,v in enumerate(VIEWS):
            expected = (images[v][:,0] > 0).long()
            expected[~present[:,i]] = 0
            torch.testing.assert_close(result[v], expected)
        model.calls = 0
        result = predict_cine(model, images, torch.tensor([[0,0,1],[0,0,1]]))
        self.assertEqual(model.calls, 1)
        self.assertEqual(set(result), {'sa'})


if __name__ == '__main__':
    unittest.main()
