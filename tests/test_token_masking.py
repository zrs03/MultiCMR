import unittest
import torch
from models.model import TwoStageAnatomyTransformer, AnatomyCrossStage


class TokenMaskingTests(unittest.TestCase):
    def setUp(self):
        self.routing = torch.tensor([[1,1,0,0,0],[1,1,1,1,1],[1,1,1,0,0]], dtype=torch.bool)

    def test_rate_single_view_missing_and_eval(self):
        net = TwoStageAnatomyTransformer(8, 2, self.routing, token_drop_rate=.05)
        present = torch.tensor([[1,1,1],[1,0,0],[0,1,1]], dtype=torch.bool)
        edges = [self.routing[i][None,None].expand(3,64,5) & present[:,i,None,None] for i in range(3)]
        masked = net.mask_anatomy_reads(edges, present)
        for i in range(3):
            for b in range(3):
                expected = 61 if present[b,i] and present[b].sum() >= 2 else (64 if present[b,i] else 0)
                self.assertEqual(masked[i][b].any(-1).sum().item(), expected)
            self.assertFalse((masked[i] & ~edges[i]).any())
        net.eval()
        for a,b in zip(net.mask_anatomy_reads(edges, present), edges):
            torch.testing.assert_close(a,b)
        net.train()
        net.token_drop_rate = 0
        self.assertIs(net.mask_anatomy_reads(edges, present), edges)
        net.token_drop_rate = .99
        for mask in net.mask_anatomy_reads([e[:,:1] for e in edges], present):
            self.assertTrue(mask[0].any())

    def test_masked_keys_do_not_affect_anatomy_but_queries_still_update(self):
        stage = AnatomyCrossStage(8,2,dropout=0).eval()
        patches = torch.randn(1,4,8,requires_grad=True)
        anatomy = torch.randn(1,5,8)
        allowed = torch.ones(1,4,5,dtype=torch.bool)
        reads = allowed.clone()
        reads[:,0] = False
        updated, summary = stage(patches, anatomy, allowed, reads)
        changed = patches.detach().clone()
        changed[:,0] += 10
        _, other = stage(changed, anatomy, allowed, reads)
        torch.testing.assert_close(summary, other)
        self.assertGreater((updated[:,0]-patches[:,0]).abs().sum().item(), 0)
        updated[:,0].square().sum().backward()
        self.assertGreater(patches.grad[:,0].abs().sum().item(), 0)

    def test_invalid_rate(self):
        for rate in (-.1,1.,float('nan')):
            with self.assertRaises(ValueError):
                TwoStageAnatomyTransformer(8,2,self.routing,token_drop_rate=rate)
