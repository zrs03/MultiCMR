import unittest
import torch
from models import MMFormerAnatomy
from utils.losses import dice_ce_loss_per_sample, masked_segmentation_loss


class LossTests(unittest.TestCase):
    def test_sample_view_weighting_and_missing_gradients(self):
        torch.manual_seed(9)
        views = ('2ch', '4ch', 'sa')
        present = torch.tensor([[1, 1, 1], [0, 0, 1]], dtype=torch.bool)
        main = {v: torch.randn(2, 3, 4, 4, requires_grad=True) for v in views}
        aux = {v: torch.randn(2, 3, 4, 4, requires_grad=True) for v in views}
        target = {v: torch.randint(3, (2, 4, 4)) for v in views}
        result = dict(logits=main, aux_logits=aux, valid_views=present)
        expected = sum(dice_ce_loss_per_sample(main[v][b:b+1], target[v][b:b+1], 3).sum()
                       + .2 * dice_ce_loss_per_sample(aux[v][b:b+1], target[v][b:b+1], 3).sum()
                       for i, v in enumerate(views) for b in range(2) if present[b, i]) / 4
        actual = masked_segmentation_loss(result, target, boundary_weight=0)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for predictions in (main, aux):
            self.assertEqual(predictions['2ch'].grad[1].abs().sum().item(), 0)
            self.assertGreater(predictions['sa'].grad[1].abs().sum().item(), 0)
        # Validation does not read auxiliary logits, even when supplied.
        for v in views:
            aux[v] = torch.full_like(aux[v], float('nan'))
        self.assertTrue(torch.isfinite(masked_segmentation_loss(
            result, target, aux_weight=0, boundary_weight=0)))

    def test_model_has_no_deep_supervision(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2, temporal_size=3)
        self.assertFalse(any('deep_heads' in key for key in model.state_dict()))
        x = {v: torch.randn(1, 1, 16 if v == 'sa' else 3, 16, 16)
             for v in ('2ch', '4ch', 'sa')}
        result = model(x, torch.ones(1, 3))
        self.assertNotIn('deep_logits', result)
        self.assertEqual(len(result['aux_logits']), 3)


if __name__ == '__main__':
    unittest.main()
