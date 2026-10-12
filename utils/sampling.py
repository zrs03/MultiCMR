"""Deterministic patient-balanced window sampling shared by all DDP ranks."""
from collections import defaultdict
import math

import torch
from torch.utils.data import Sampler


class PatientWindowSampler(Sampler):
    def __init__(self, cases, windows_per_patient=4, seed=42, rank=0, num_replicas=1):
        if windows_per_patient < 0 or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError('Invalid window count or distributed rank')
        self.groups = defaultdict(list)
        for index, case in enumerate(cases):
            self.groups[case.case_id].append(index)
        if not self.groups:
            raise ValueError('No training patients')
        self.windows_per_patient = windows_per_patient
        self.seed, self.rank, self.num_replicas = seed, rank, num_replicas
        self.epoch = 0
        self.selected_count = sum(min(len(g), windows_per_patient) if windows_per_patient else len(g)
                                  for g in self.groups.values())
        self.num_samples = math.ceil(self.selected_count / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = []
        for patient in sorted(self.groups):
            group = self.groups[patient]
            count = min(len(group), self.windows_per_patient) if self.windows_per_patient else len(group)
            selected.extend(group[i] for i in torch.randperm(len(group), generator=generator)[:count].tolist())
        selected = [selected[i] for i in torch.randperm(len(selected), generator=generator).tolist()]
        # Equal numbers of batches are required by DDP. Only the minimal padding
        # may duplicate samples; sampling within each patient's quota is unique.
        padding = self.total_size - len(selected)
        selected += (selected * math.ceil(padding / len(selected)))[:padding]
        return iter(selected[self.rank:self.total_size:self.num_replicas])
