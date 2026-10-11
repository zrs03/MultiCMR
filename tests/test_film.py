"""FiLM identity initialization, anatomy routing and spatial broadcast."""
import unittest

import torch

from models.model import AnatomyFiLM, MMFormerAnatomy


class FiLMTests(unittest.TestCase):
    def test_identity_and_channel_broadcast_2d_3d(self):
        for spatial in ((6, 7), (4, 6, 7)):
            film = AnatomyFiLM(3, 8, 2)
            x = torch.randn(2, 3, *spatial, requires_grad=True)
            anatomy = torch.randn(2, 5, 8)
            available = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
            torch.testing.assert_close(film(x, anatomy, available), x, rtol=0, atol=0)
            film(x, anatomy, available).square().mean().backward()
            self.assertGreater(film.generator[-1].weight.grad.abs().sum().item(), 0)
            with torch.no_grad():
                film.generator[-1].bias.copy_(torch.tensor([.1, .2, .3, 1., 2., 3.]))
            shape = (1, 3) + (1,) * len(spatial)
            expected = x * torch.tensor([1.1, 1.2, 1.3]).reshape(shape) + torch.tensor([1., 2., 3.]).reshape(shape)
            torch.testing.assert_close(film(x, anatomy, available), expected)

    def test_only_selected_tokens_condition_film(self):
        torch.manual_seed(7)
        film = AnatomyFiLM(3, 8, 2)
        torch.nn.init.normal_(film.generator[-1].weight, std=.1)
        x = torch.randn(2, 3, 6, 7)
        anatomy = torch.randn(2, 5, 8, requires_grad=True)
        available = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
        expected = film(x, anatomy, available)
        changed = anatomy.detach().clone()
        changed[:, 2:] = float('nan')
        torch.testing.assert_close(film(x, changed, available), expected)
        expected.square().mean().backward()
        self.assertGreater(anatomy.grad[:, :2].abs().sum().item(), 0)
        self.assertEqual(anatomy.grad[:, 2:].abs().sum().item(), 0)

    def test_generators_are_independent_per_view_and_scale(self):
        model = MMFormerAnatomy(base=2, dim=8, heads=2)
        modules = []
        for scale in ('a1', 'a2', 'a3'):
            lax = getattr(model.decoder_2d, scale)
            sax = getattr(model.decoder, scale)
            self.assertEqual(lax['2ch'].num_tokens, 2)
            self.assertEqual(lax['4ch'].num_tokens, 5)
            self.assertEqual(sax['sa'].num_tokens, 3)
            modules.extend([lax['2ch'], lax['4ch'], sax['sa']])
        self.assertEqual(len({id(m.generator[-1].weight) for m in modules}), 9)


if __name__ == '__main__':
    unittest.main()
