import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_dataset(dataset_to_get):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    if dataset_to_get == 'mnist':
        train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset_to_get == 'cifar10':
        train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset_to_get))

    dataset = {'name': dataset_to_get, 'train': train, 'test': test}

    return dataset
