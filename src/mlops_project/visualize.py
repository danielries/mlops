import torch
import typer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize(model_checkpoint: str, figure_name: str = "embeddings") -> None:
    """Visualize the training statistics."""
    model = torch.load(model_checkpoint)

    model.eval()

    model.fc = torch.nn.Identity()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings_list = []
    targets_list = []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=32):
            images, targets = batch
            embeddings_list.append(model(images).detach())
            targets_list.append(targets)

        embeddings = torch.cat(embeddings_list)
        targets = torch.cat(targets_list)

    embeddings = TSNE(n_components=2).fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        indices = targets == i
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}.png")
    print("Visualization complete")


if __name__ == "__main__":
    typer.run(visualize)
