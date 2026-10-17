import unittest
import torch
from models.model import CineViewEncoder, MMFormerAnatomy


class ChannelWindowTests(unittest.TestCase):
    def test_default_token_count_and_window_skips(self):
        encoder = CineViewEncoder(1, 2, 8, (2,4,4), 0, 2, temporal_size=5).eval()
        x = torch.randn(1,1,5,128,128)
        changed = x.clone()
        changed[:, :, 0] += 2
        with torch.no_grad():
            skips, tokens, shape = encoder(x)
            other_skips, other_tokens, _ = encoder(changed)
        self.assertEqual(shape, (8,8))
        self.assertEqual(tokens.shape, (1,64,8))
        for a,b in zip(skips, other_skips):
            self.assertGreater((a-b).abs().sum().item(), 0)
        self.assertGreater((tokens[:,:64]-other_tokens[:,:64]).abs().sum().item(), 0)

    def test_center_output_backpropagates_to_neighbor_frames(self):
        torch.manual_seed(7)
        model = MMFormerAnatomy(base=2, dim=8, heads=2, temporal_size=3).eval()
        images = {v: torch.randn(1,1,16 if v=='sa' else 3,16,16, requires_grad=True)
                  for v in ('2ch','4ch','sa')}
        result = model(images, torch.tensor([[1,0,1]]))
        self.assertEqual(result['logits']['2ch'].shape, (1,3,16,16))
        result['logits']['2ch'].square().mean().backward()
        for t in (0,1,2):
            self.assertGreater(images['2ch'].grad[:,:,t].abs().sum().item(), 0)
        grad = model.encoders['2ch'].c1.conv[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        self.assertIsNone(images['4ch'].grad)


if __name__ == '__main__':
    unittest.main()
