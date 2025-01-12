import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    model = torch.load(model_checkpoint).to(DEVICE)
    model.eval()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in test_dataloader:
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f"Accuracy: {correct / total}")


if __name__ == "__main__":
    typer.run(evaluate)
