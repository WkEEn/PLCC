# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn


class PLCC(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, num_classes, hidden_dim=2048, pred_dim=256, momentum=0.99):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(PLCC, self).__init__()

        self.momentum = momentum
        # create the online encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.online_encoder = base_encoder(num_classes=pred_dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.online_encoder.fc.weight.shape[1]
        self.online_encoder.fc = nn.Sequential(nn.Linear(prev_dim, hidden_dim, bias=False),
                                               nn.BatchNorm1d(hidden_dim),
                                               nn.ReLU(inplace=True),  # hidden layer
                                               nn.Linear(hidden_dim, pred_dim))  # output layer

        # create the target encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for target_weight in self.target_encoder.parameters():
            target_weight.requires_grad = False

        # build a classifier
        self.predictor = nn.Sequential(nn.Linear(pred_dim, hidden_dim, bias=False),
                                       nn.BatchNorm1d(hidden_dim),
                                       nn.ReLU(inplace=True),  # hidden layer
                                       nn.Linear(hidden_dim, pred_dim))  # output layer

        self.fc = nn.Linear(pred_dim, num_classes)

    @torch.no_grad()
    def _update_moving_average(self):
        for online_encoder, target_encoder in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_encoder.data = target_encoder.data * self.momentum + online_encoder.data * (1. - self.momentum)

    def forward(self, x):
        """
        Input:
            x: augmented views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for input
        online_z = self.online_encoder(x)  # NxC
        online_p = self.predictor(online_z)

        with torch.no_grad():
            self._update_moving_average()
            target_z = self.target_encoder(x)  # NxC

        return self.fc(online_p), online_p, target_z.detach()
