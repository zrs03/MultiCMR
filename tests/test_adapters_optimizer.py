import tempfile
import unittest
from pathlib import Path

import torch

from models.model import MMFormerAnatomy, ViewSpatialAdapter
from utils.optimization import build_adamw
from utils.checkpoint import capture_rng_state, restore_training_state


class AdapterOptimizerTests(unittest.TestCase):
    def test_identity_initialization_and_residual_gradients(self):
        adapter = ViewSpatialAdapter(4)
        x = torch.randn(2, 4, 9, 11, requires_grad=True)
        torch.testing.assert_close(adapter(x), x, rtol=0, atol=0)
        optimizer = build_adamw(adapter)
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.pointwise.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad()
        adapter(x).square().mean().backward()
        self.assertGreater(adapter.depthwise.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_view_and_scale_adapters_are_independent(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2)
        adapters = model.decoder_2d.spatial_adapters
        self.assertEqual(set(adapters), {'2ch', '4ch'})
        self.assertEqual(len(model.decoder.spatial_adapters), 0)
        params = [m.pointwise.weight for modules in adapters.values() for m in modules]
        self.assertEqual(len({id(p) for p in params}), 4)
        with torch.no_grad():
            adapters['2ch'][1].pointwise.bias.fill_(1.)
        x = torch.randn(1, 2, 9, 11)
        torch.testing.assert_close(adapters['2ch'][1](x), x + 1)
        torch.testing.assert_close(adapters['4ch'][1](x), x)

    def test_adamw_groups_and_resume(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2)
        optimizer = build_adamw(model, weight_decay=.001)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        groups = {id(p): g['weight_decay'] for g in optimizer.param_groups for p in g['params']}
        self.assertEqual(len(groups), len(list(model.parameters())))
        self.assertEqual(groups[id(model.encoders['2ch'].c1.conv[0].weight)], .001)
        self.assertEqual(groups[id(model.inter.lax_stage.anatomy_reads.attn.in_proj_weight)], .001)
        self.assertEqual(groups[id(model.inter.anatomy_tokens)], 0)
        self.assertEqual(groups[id(model.context_scale)], 0)
        self.assertEqual(groups[id(model.encoders['2ch'].token_pos)], 0)
        self.assertEqual(groups[id(model.decoder_2d.spatial_adapters['2ch'][0].norm.weight)], 0)
        # Exercise state restoration with both AdamW parameter groups populated.
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        scheduler.step(1.)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'state.pth'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=2, best_val_dice=.8,
                            rng_states=[capture_rng_state()]), path)
            restored = MMFormerAnatomy(base=2, dim=8, heads=2)
            opt = build_adamw(restored, lr=.1, weight_decay=.1)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)
            self.assertEqual(restore_training_state(torch.load(path, weights_only=False), restored,
                                                    opt, sched), (3, .8))
            self.assertEqual([g['weight_decay'] for g in opt.param_groups], [.001, 0])
            self.assertEqual(opt.param_groups[0]['lr'], 1e-4)
            for p, q in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(p, q)
                torch.testing.assert_close(optimizer.state[p]['exp_avg'], opt.state[q]['exp_avg'])


if __name__ == '__main__':
    unittest.main()
