"""Machine learning models"""
from .FPLNet import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    FPLNet,
    FPLMultiTaskNet,
    UncertaintyWeightedLoss,
    FPLNetTrainer,
    FPLMultiTaskTrainer,
)

from .Trainer import NNTrainer
from .MLP import MLP
from .CNN import CNN



__all__ = [
    'GatedResidualNetwork',
    'VariableSelectionNetwork',
    'FPLNet',
    'FPLMultiTaskNet',
    'UncertaintyWeightedLoss',
    'FPLNet',
    'FPLNetTrainer',       
    'FPLMultiTaskTrainer',
    'NNTrainer',
    'MLP',
    'CNN',
]