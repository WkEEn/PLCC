import math
from PIL import Image

import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Sampler


class CategoryDataset(data.Dataset):
    def __init__(self, data_path, dataset, num_classes, transforms):
        data_file = open(data_path)

        self.transforms = transforms
        if dataset == 'HER2':
            from masks.her2_mask import MASK
            self.mask = MASK
        if dataset == 'egfr':
            from masks.egfr_mask import MASK
            self.mask = MASK
        self.num_classes = num_classes

        self.images = []
        self.labels = []
        try:
            text_lines = data_file.readlines()
            for i in text_lines:
                i = i.strip()
                self.images.append(i.split(' ')[0])
                self.labels.append(int(i.split(' ')[1]))
        finally:
            data_file.close()

    def __getitem__(self, ind):
        image = Image.open(self.images[ind])
        image = self.transforms(image)

        label = self.labels[ind]

        mask = torch.tensor(self.mask[label])

        return image, label, mask, ind

    def __len__(self):
        return len(self.images)


class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True, shuffle=True, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement
        self.shuffle = shuffle
        self.drop_last = drop_last

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        # targets = torch.Tensor(self.dataset.labels).long()
        targets = torch.Tensor(self.dataset.targets).long()
        self.weights = self.calculate_weights(targets)

    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double() * targets.numel()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            self.weights = self.weights[indices]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            self.weights = torch.cat((self.weights, self.weights[:(self.total_size - len(indices))]))
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
            self.weights = self.weights[:self.total_size]
        assert len(indices) == self.total_size and len(self.weights) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(torch.multinomial(self.weights[indices], self.num_samples, self.replacement).tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
