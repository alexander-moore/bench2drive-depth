"""
Abstract base class for all depth models.
"""
from argparse import ArgumentParser, Namespace

import torch.nn as nn


class DepthModelBase(nn.Module):
    """
    ABC for all depth estimation models in this project.

    Subclasses must set the following class attributes and implement the
    three methods below.
    """

    produces_semantic: bool = False
    is_stateful:       bool = False
    state_keys:        tuple = ()   # e.g. ("depth_tokens",)

    def forward(self, x, state=None):
        """
        Args:
            x     : (B, S, C, 3, H, W)
            state : dict of tensors (stateful models) or None

        Returns:
            ModelOutput with at least .depth filled
        """
        raise NotImplementedError

    @classmethod
    def add_model_args(cls, parser: ArgumentParser) -> None:
        """Add model-specific CLI flags to an existing ArgumentParser."""
        pass

    @classmethod
    def from_args(cls, args: Namespace):
        """Construct the model from parsed args."""
        raise NotImplementedError
