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
        deep = {v: torch.randn(2, 3, 4, 4, requires_grad=True) for v in views}
        target = {v: torch.randint(3, (2, 4, 4)) for v in views}
        result = dict(logits=main, aux_logits=aux, deep_logits=deep, valid_views=present)
        expected = sum(dice_ce_loss_per_sample(main[v][b:b+1], target[v][b:b+1], 3).sum()
                       + .2 * dice_ce_loss_per_sample(aux[v][b:b+1], target[v][b:b+1], 3).sum()
                       + .1 * dice_ce_loss_per_sample(deep[v][b:b+1], target[v][b:b+1], 3).sum()
                       for i, v in enumerate(views) for b in range(2) if present[b, i]) / 4
        actual = masked_segmentation_loss(result, target, boundary_weight=0)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        for predictions in (main, aux, deep):
            self.assertEqual(predictions['2ch'].grad[1].abs().sum().item(), 0)
            self.assertGreater(predictions['sa'].grad[1].abs().sum().item(), 0)
        # Validation does not read auxiliary logits, even when supplied.
        for v in views:
            aux[v] = torch.full_like(aux[v], float('nan'))
            deep[v] = torch.full_like(deep[v], float('nan'))
        self.assertTrue(torch.isfinite(masked_segmentation_loss(
            result, target, aux_weight=0, deep_weight=0, boundary_weight=0)))

    def test_single_scale_training_only_and_gradients(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2, temporal_size=3)
        self.assertEqual(len(model.deep_heads), 3)
        x = {v: torch.randn(1, 1, 5 if v == 'sa' else 3, 16, 16)
             for v in ('2ch', '4ch', 'sa')}
        result = model(x, torch.ones(1, 3))
        self.assertEqual(set(result['deep_logits']), {'2ch', '4ch', 'sa'})
        for v in result['logits']:
            self.assertEqual(result['deep_logits'][v].shape, result['logits'][v].shape)
        targets = {v: torch.randint(t.shape[1], (t.shape[0], *t.shape[2:]))
                   for v,t in result['logits'].items()}
        without = masked_segmentation_loss(result, targets, boundary_weight=0, deep_weight=0)
        with_deep = masked_segmentation_loss(result, targets, boundary_weight=0, deep_weight=.1)
        expected = sum(dice_ce_loss_per_sample(t, targets[v], t.shape[1]).mean()
                       for v,t in result['deep_logits'].items()) / 3
        torch.testing.assert_close(with_deep - without, .1 * expected)
        with_deep.backward()
        for head in model.deep_heads.values():
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)
        model.eval()
        calls = []
        hooks = [head.register_forward_hook(lambda *args: calls.append(1)) for head in model.deep_heads.values()]
        with torch.no_grad():
            evaluated = model(x, torch.ones(1, 3))
        for hook in hooks:
            hook.remove()
        self.assertEqual(evaluated['deep_logits'], {})
        self.assertEqual(calls, [])
        self.assertEqual(len(result['aux_logits']), 3)


if __name__ == '__main__':
    unittest.main()
