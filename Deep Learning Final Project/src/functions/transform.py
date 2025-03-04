# define the transformation steps
import torchvision.transforms as transforms


def transformer():
    # Define transformations (data augmentation and normalization)
    transform = transforms.Compose([
        transforms.RandomRotation(),  # Randomly flip the image
        transforms.RandomCrop(32, padding=4),  # Random crop
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),  # Gaussian blur
        transforms.ToTensor(),  # Convert the image to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])
    return transform
