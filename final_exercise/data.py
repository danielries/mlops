import torch
import matplotlib.pyplot as plt


DATA_PATH = "data"

def corrupt_mnist():
    """Return train and test dataloaders for corrupt MNIST."""
    # exchange with the corrupted mnist dataset
    
    # Load the training images and targets from the 6 files
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load the test images and targets
    test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

    # Reshape the images to have a single channel
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Create the datasets
    train = torch.utils.data.TensorDataset(train_images, train_target)
    test = torch.utils.data.TensorDataset(test_images, test_target)

    return train, test


def show_image_and_target(images: torch.Tensor, target: torch.Tensor):
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)
    fig, axes = plt.subplots(row_col, row_col, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze(), cmap="gray")
        ax.set_title(f"Label: {target[i].item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    train, test = corrupt_mnist()
    print(f"Size of training set: {len(train)}")
    print(f"Size of test set: {len(test)}")
    print(f"Shape of a training point {(train[0][0].shape, train[0][1].shape)}")
    print(f"Shape of a test point {(test[0][0].shape, test[0][1].shape)}")
    show_image_and_target(train.tensors[0][:25], train.tensors[1][:25])


