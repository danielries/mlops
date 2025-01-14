import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """
    Train a model on MNIST.

    Args:
        lr (float): Learning rate for the optimizer.
        batch_size (int): Number of samples per batch.
        epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Initialize the model and move it to the specified device
    model = MyAwesomeModel().to(DEVICE)

    # Load the training dataset
    train_set, _ = corrupt_mnist()

    # Create a DataLoader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Define the loss function and the optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dictionary to store training statistics
    statistics = {"train_loss": [], "train_accuracy": []}

    # Training loop
    for epoch in range(epochs):
        model.train()
        for images, targets in train_dataloader:
            optimizer.zero_grad()  # Zero the gradients
            output = model(images)  # Forward pass
            loss = criterion(output, targets)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

            # Store the loss and accuracy
            statistics["train_loss"].append(loss.item())
            accuracy = (output.argmax(dim=1) == targets).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

        print(f"Epoch {epoch}, loss: {loss.item()}")

    # Save the trained model
    torch.save(model, "models/model.pth")

    # Plot and save training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

    print("Training complete")


if __name__ == "__main__":
    typer.run(train)
