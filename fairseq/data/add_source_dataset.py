# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple

import torch

from fairseq.data import (
    data_utils as fairseq_data_utils,
)
from . import BaseWrapperDataset, data_utils


def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


class AddSourceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_sources,
        data_cfg=None,
        process_label=None,
        add_to_input=False,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_sources = batch_sources
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input
        self.data_cfg = data_cfg

    def get_label(self, index):
        return (
            self.labels[index]
            if self.process_label is None
            else self.process_label(self.labels[index])
        )

    def __getitem__(self, index):
        item = self.dataset[index]
        # item["label"] = self.get_label(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        target_lengths = torch.tensor(
            [len(self.get_label(index)) for index in range(len(samples))],
            dtype=torch.long,
        )
        collated["net_input"]["target_lengths"] = target_lengths
        return collated
