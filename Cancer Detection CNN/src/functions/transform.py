# define the transformation steps
import torchvision.transforms as transforms


def transformer():
    # Define transformations (data augmentation and normalization)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image
        transforms.ColorJitter(),  # Randomly change the brightness, contrast, and saturation of an image
        transforms.RandomCrop(32, padding=4),  # Random crop
        transforms.ToTensor(),  # Convert the image to PyTorch tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
    ])
    return transform
