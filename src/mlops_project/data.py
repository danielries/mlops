import torch
import typer


RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def normalize(images: torch.Tensor):
    """Normalize to mean 0 and std 1."""
    return (images - images.mean()) / images.std()

def pre_process_mnist(raw_path: str = RAW_DATA_PATH, processed_path: str = PROCESSED_DATA_PATH):
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    
    # Load the training images and targets from the 6 files
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{RAW_DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{RAW_DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load the test images and targets
    test_images: torch.Tensor = torch.load(f"{RAW_DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{RAW_DATA_PATH}/test_target.pt")

    # Reshape the images to have a single channel
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Create the datasets
    train = torch.utils.data.TensorDataset(train_images, train_target)
    test = torch.utils.data.TensorDataset(test_images, test_target)

    # Save the datasets
    torch.save(train_images, f"{PROCESSED_DATA_PATH}/train_images.pt")
    torch.save(train_target, f"{PROCESSED_DATA_PATH}/train_target.pt")
    torch.save(test_images, f"{PROCESSED_DATA_PATH}/test_images.pt")
    torch.save(test_target, f"{PROCESSED_DATA_PATH}/test_target.pt")

    return train, test

def corrupt_mnist():
    "Load the processed corrupted MNIST datasets."
    train_images = torch.load(f"{PROCESSED_DATA_PATH}/train_images.pt")
    train_target = torch.load(f"{PROCESSED_DATA_PATH}/train_target.pt")
    test_images = torch.load(f"{PROCESSED_DATA_PATH}/test_images.pt")
    test_target = torch.load(f"{PROCESSED_DATA_PATH}/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(pre_process_mnist)
