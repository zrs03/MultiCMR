import unittest
import torch

from models.model import TransformerBlock
from utils.checkpoint import capture_rng_state, restore_training_state
from utils.inference import predict_cine


class FP16Tests(unittest.TestCase):
    def test_cpu_fp16_rejected(self):
        images = {v: torch.zeros(1, 1, 3, 8, 8) for v in ('2ch', '4ch', 'sa')}
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            predict_cine(torch.nn.Identity().eval(), images, torch.ones(1, 3), fp16=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_chunked_attention_fp16_gradients_and_scaler_resume(self):
        torch.manual_seed(42)
        model = TransformerBlock(16, 4, dropout=0, query_chunk_size=4).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        make_scaler = (lambda: torch.amp.GradScaler('cuda', init_scale=128)) if hasattr(torch.amp, 'GradScaler') else (lambda: torch.cuda.amp.GradScaler(init_scale=128))
        scaler = make_scaler()
        x = torch.randn(2, 12, 16, device='cuda')
        missing = torch.zeros(2, 12, dtype=torch.bool, device='cuda')
        missing[0, :4] = True
        before = model.attn.in_proj_weight.detach().clone()
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(x, key_padding_mask=missing)
        loss = output.float().square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        scaler.step(optimizer)
        scaler.update()
        self.assertFalse(torch.equal(before, model.attn.in_proj_weight))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     epoch=1, best_val_dice=.5, rng_states=[capture_rng_state()])
        restored = make_scaler()
        self.assertEqual(restore_training_state(state, model, optimizer, scheduler, scaler=restored), (2, .5))
        self.assertEqual(restored.state_dict(), scaler.state_dict())
        del state['scaler']
        with self.assertWarnsRegex(UserWarning, 'fresh GradScaler'):
            restore_training_state(state, model, optimizer, scheduler, scaler=make_scaler())


if __name__ == '__main__':
    unittest.main()
