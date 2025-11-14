from .evaluate_lenet import evaluate as evaluate_lenet
from .evaluate_model import evaluate_classifier
from .evaluate_alexnet import evaluate as evaluate_alexnet

__all__ = ['evaluate_lenet', 'evaluate_classifier', 'evaluate_alexnet']