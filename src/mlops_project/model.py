from torch import nn
import torch


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        """
        Initialize the model layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # First convolutional layer
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(64, 128, 3, 1)  # Third convolutional layer
        self.conv4 = nn.Conv2d(128, 256, 3, 1)  # Added fourth convolutional layer
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.fc1 = nn.Linear(256, 10)  # Fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = torch.relu(self.conv1(x))  # Apply ReLU activation to the first conv layer
        x = torch.max_pool2d(x, 2, 2)  # Apply max pooling
        x = torch.relu(self.conv2(x))  # Apply ReLU activation to the second conv layer
        x = torch.max_pool2d(x, 2, 2)  # Apply max pooling
        x = torch.relu(self.conv3(x))  # Apply ReLU activation to the third conv layer
        x = torch.max_pool2d(x, 2, 2)  # Apply max pooling
        x = torch.relu(self.conv4(x))  # Apply ReLU activation to the fourth conv layer
        x = torch.max_pool2d(x, 2, 2)  # Apply max pooling
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout
        return self.fc1(x)  # Apply the fully connected layer


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
