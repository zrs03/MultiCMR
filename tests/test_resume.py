"""Checkpoint continuation must restore optimizer, schedule and random streams."""
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.checkpoint import atomic_save, capture_rng_state, restore_training_state


class ResumeTests(unittest.TestCase):
    def test_next_step_matches_uninterrupted_training(self):
        torch.manual_seed(12)
        random.seed(12)
        np.random.seed(12)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0)

        def step(net, opt):
            x = torch.randn(5, 3) * np.random.rand() + random.random()
            opt.zero_grad()
            loss = net(x).square().mean()
            loss.backward()
            opt.step()
            return loss.detach()

        step(model, optimizer)
        scheduler.step(1.)
        scheduler.step(2.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'last.pth'
            atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), epoch=7, best_val_dice=.84,
                             rng_states=[capture_rng_state()]), path)
            expected = step(model, optimizer)
            resumed = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.3), torch.nn.Linear(4, 1))
            opt = torch.optim.Adam(resumed.parameters(), lr=.9)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20)
            epoch, best = restore_training_state(torch.load(path, weights_only=False), resumed, opt, sched)
            self.assertEqual((epoch, best), (8, .84))
            self.assertEqual(opt.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
            self.assertEqual(sched.state_dict(), scheduler.state_dict())
            actual = step(resumed, opt)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for a, b in zip(resumed.parameters(), model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(Path(str(path) + '.tmp').exists())

    def test_old_weights_warn_and_start_at_one(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        with self.assertWarnsRegex(UserWarning, 'weights only'):
            self.assertEqual(restore_training_state({'model': model.state_dict()}, model,
                                                    optimizer, scheduler), (1, -1.))

    def test_world_size_change_warns_but_restores_epoch(self):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          scheduler=scheduler.state_dict(), epoch=2, best_val_dice=.5,
                          rng_states=[capture_rng_state()])
        with self.assertWarnsRegex(UserWarning, 'world size changed'):
            self.assertEqual(restore_training_state(checkpoint, model, optimizer, scheduler,
                                                    world_size=2), (3, .5))


if __name__ == '__main__':
    unittest.main()
