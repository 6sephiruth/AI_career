
import torchvision.datasets as dataset
import torchvision.transforms as transforms

class mnist():
    def __init__(self):
    
        self.train = dataset.MNIST(root='dataset/mnist',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

        self.test = dataset.MNIST(root='dataset/mnist',
                                train=False,
                                transform=transforms.ToTensor(),
                                download=True)