"""Anatomical edge restrictions, global exchange and chunked masked attention."""
import unittest

import torch

from models.model import MMFormerAnatomy, TwoStageAnatomyTransformer, TransformerBlock


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.available = torch.tensor([[1,1,0,0,0], [1,1,1,1,1], [1,1,1,0,0]], dtype=torch.bool)

    def test_routing_edges(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available)
        mask = net.routing_mask([2, 3, 4], torch.device('cpu'))
        self.assertFalse(mask[:2, :2].any())
        self.assertTrue(mask[:2, 2:9].all())
        torch.testing.assert_close(mask[0, 9:], ~self.available[0])
        torch.testing.assert_close(mask[5, 9:], ~self.available[2])
        self.assertTrue(mask[12, :2].all())  # RA cannot read 2CH.
        self.assertFalse(mask[12, 2:5].any())  # RA can read 4CH.
        self.assertTrue(mask[12, 5:9].all())  # RA cannot read SAX.
        self.assertFalse(mask[12, 12])
        self.assertTrue(mask[12, 13])  # Different anatomy tokens only mix in stage 2.
        self.assertFalse(mask[9, :9].any())  # Myocardium reads all views.

    def test_first_stage_is_restricted_second_stage_is_global(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available, dropout=0).eval()
        tokens = {v: torch.randn(1, 2, 8) for v in ('2ch', '4ch', 'sa')}
        present = torch.ones(1, 3, dtype=torch.bool)
        routed_outputs = []
        hook = net.routed_layer.register_forward_hook(lambda m, a, output: routed_outputs.append(output.detach()))
        _, first = net(tokens, present)
        changed = {**tokens, '2ch': torch.randn_like(tokens['2ch']) * 4}
        _, second = net(changed, present)
        hook.remove()
        # RA (anatomy index 3) sees no 2CH in the first simultaneous update.
        torch.testing.assert_close(routed_outputs[0][:, 9], routed_outputs[1][:, 9], rtol=0, atol=0)
        self.assertGreater((first[:, 3] - second[:, 3]).abs().max().item(), 1e-6)
        # Missing 2CH never affects either stage, even if its inputs are NaN.
        present[:, 0] = False
        expected = net(tokens, present)
        changed['2ch'].fill_(float('nan'))
        actual = net(changed, present)
        torch.testing.assert_close(expected[1], actual[1])
        for view in tokens:
            torch.testing.assert_close(expected[0][view], actual[0][view])

    def test_chunked_routing_and_padding_match_full_attention_gradients(self):
        net = TwoStageAnatomyTransformer(8, 2, self.available)
        routing = net.routing_mask([2, 3, 4], torch.device('cpu'))
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
