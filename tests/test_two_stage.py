"""Sequential cross-attention routing, missing-view isolation and intra attention."""
import unittest

import torch

from models.model import MMFormerAnatomy, TwoStageAnatomyTransformer, TransformerBlock


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.available = torch.tensor([[1,1,0,0,0], [1,1,1,1,1], [1,1,1,0,0]], dtype=torch.bool)

    def test_routing_edges_and_stage_order(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available, dropout=0).eval()
        tokens = {v: torch.randn(1, 2, 8) for v in ('2ch', '4ch', 'sa')}
        present = torch.ones(1, 3, dtype=torch.bool)
        captured = []
        hook = net.lax_stage.register_forward_hook(lambda m, a, out: captured.append((a[2], out[1])))
        first_views, first = net(tokens, present)
        changed = {**tokens, '2ch': torch.randn_like(tokens['2ch']) * 4}
        _, second = net(changed, present)
        hook.remove()
        torch.testing.assert_close(captured[0][0][0, 0], self.available[0])
        torch.testing.assert_close(captured[0][0][0, 2], self.available[1])
        # RA/LA can now receive 2CH evidence indirectly through updated 4CH patches.
        self.assertGreater((first_views['4ch'] - net(changed, present)[0]['4ch']).abs().sum().item(), 0)
        self.assertGreater((first[:, :2] - second[:, :2]).abs().max().item(), 1e-6)
        changed = {**tokens, 'sa': torch.randn_like(tokens['sa']) * 4}
        other_views, other = net(changed, present)
        # SAX is processed later: no backward patch update to LAX in this block.
        torch.testing.assert_close(first_views['2ch'], other_views['2ch'])
        torch.testing.assert_close(first_views['4ch'], other_views['4ch'])
        torch.testing.assert_close(first[:, 3:], other[:, 3:])
        self.assertGreater((first[:, :3]-other[:, :3]).abs().sum().item(), 0)

    def test_joint_lax_attention_is_bidirectional_and_excludes_sax(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available, dropout=0).eval()
        tokens = {v: torch.randn(1, 4, 8) for v in ('2ch', '4ch', 'sa')}
        captured = []
        hook = net.lax_patch_attention.register_forward_hook(
            lambda m, args, output: captured.append(output.detach().clone()))
        present = torch.ones(1, 3, dtype=torch.bool)
        net(tokens, present)
        for view in ('2ch', '4ch', 'sa'):
            changed = {**tokens, view: torch.randn_like(tokens[view])}
            net(changed, present)
        hook.remove()
        self.assertGreater((captured[0][:,4:] - captured[1][:,4:]).abs().sum().item(), 0)
        self.assertGreater((captured[0][:,:4] - captured[2][:,:4]).abs().sum().item(), 0)
        torch.testing.assert_close(captured[0], captured[3])

    def test_all_missing_patterns_are_finite_and_isolated(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available, dropout=0).eval()
        present = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]], dtype=torch.bool)
        tokens = {v: torch.randn(7, 3, 8, requires_grad=True) for v in ('2ch','4ch','sa')}
        expected = net(tokens, present)
        corrupted = {v: t.detach().masked_fill(~present[:,i,None,None], float('nan'))
                     for i,(v,t) in enumerate(tokens.items())}
        actual = net(corrupted, present)
        torch.testing.assert_close(expected[1], actual[1])
        for v in tokens:
            torch.testing.assert_close(expected[0][v], actual[0][v])
        (expected[1] * torch.randn_like(expected[1])).sum().backward()
        for i,t in enumerate(tokens.values()):
            self.assertTrue(torch.isfinite(t.grad).all())
            self.assertEqual(t.grad[~present[:,i]].abs().sum().item(), 0)
        # With SAX absent, its stage leaves anatomy exactly unchanged.
        anatomy = torch.randn(1,5,8)
        _, unchanged = net.sax_stage(torch.zeros(1,3,8), anatomy, torch.zeros(1,3,5,dtype=torch.bool))
        torch.testing.assert_close(unchanged, anatomy, rtol=0, atol=0)

    def test_chunked_routing_and_padding_match_full_attention_gradients(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available)
        routing = torch.zeros(14, 14, dtype=torch.bool)
        routing[:, 1:3] = True
        full = TransformerBlock(8, 2, dropout=0, query_chunk_size=100)
        chunked = TransformerBlock(8, 2, dropout=0, query_chunk_size=3)
        chunked.load_state_dict(full.state_dict())
        x = torch.randn(2, 14, 8, requires_grad=True)
        xc = x.detach().clone().requires_grad_()
        missing = torch.zeros(2, 14, dtype=torch.bool)
        missing[0, 2:9] = True  # Only 2CH: RV/atria still have a valid self key.
        a = full(x, routing, missing)
        b = chunked(xc, routing, missing)
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        b.square().mean().backward()
        torch.testing.assert_close(x.grad, xc.grad)
        for p, q in zip(full.parameters(), chunked.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)

    def test_no_fusion_parameters_or_outputs(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2, temporal_size=3).eval()
        self.assertFalse(any('fusion' in name for name, _ in model.named_parameters()))
        x = {v: torch.randn(1, 1, 16 if v == 'sa' else 3, 16, 16)
             for v in ('2ch', '4ch', 'sa')}
        with torch.no_grad():
            out = model(x, torch.ones(1, 3))
        self.assertNotIn('fusion_tokens', out)
        self.assertEqual(out['anatomy_tokens'].shape, (1, 5, 8))
        self.assertEqual(sum(t.shape[1] for t in out['pred_tokens'].values()) + 5, 21)
        with self.assertRaises(ValueError):
            MMFormerAnatomy(inter_layers=1)


if __name__ == '__main__':
    unittest.main()
