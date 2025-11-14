from .fashion_mnist import fashion_mnist_transform
from .imagenette import imagenette_train_transform, imagenette_test_transform
from .mnist import mnist_transform

__all__ = ['fashion_mnist_transform', 'imagenette_train_transform', 'imagenette_test_transform', 'mnist_transform']