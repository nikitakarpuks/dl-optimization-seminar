import torchvision.datasets as datasets
import torchvision.transforms as transforms


mnist = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
cifar10 = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())


