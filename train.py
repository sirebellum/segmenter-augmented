import sys

# Imports neded for torch training
import torch
import torch.nn as nn
import torch.optim as optim

# Imports needed for data loading
from torch.utils.data import DataLoader
from land_coverage import LandCoverageDataset

# Imports needed for training loop
from tqdm import tqdm

# Import model
from model import AE, TILE_SIZE, NUM_SEGMENTS

import cv2
import numpy as np
import glob

batch_size = 8

# Detach function
def detach(x):
    return x.cpu().detach().numpy()

# main training function
def train(pixel_size):
    # Create model
    model = AE(
        input_shape=(TILE_SIZE, TILE_SIZE),
        pixel_size=pixel_size,
        in_channels=3,
        vectors=NUM_SEGMENTS,
        decode=True,
    ).cuda()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Create loss function
    class AugmentedLoss(nn.Module):
        def __init__(self):
            super(AugmentedLoss, self).__init__()
            self.mse_loss = nn.MSELoss()
            self.cel_loss = nn.CrossEntropyLoss()

        def forward(self, y_hat, y, x_hat, x):
            # Compute mse loss
            mse = self.mse_loss(y_hat, y)

            # Softmax the clusters
            x_hat = torch.nn.functional.softmax(x_hat, dim=1)

            # Upsample the clusters to match the original coverage
            x_hat = nn.Upsample(size=(TILE_SIZE, TILE_SIZE))(x_hat)

            # Get rid of the depth dimension
            x_hat = torch.squeeze(x_hat, dim=1)
            x = torch.squeeze(x, dim=1)

            # Compute the CEL between the clusters and the original coverage
            cel = self.cel_loss(x_hat, x)

            return mse + cel
    augmented_loss = AugmentedLoss()

    # Iterate through the maps and coverages, n times
    dataset = LandCoverageDataset(
        tile_size=TILE_SIZE,
        scale=1,
    )

    # Create a dataloader object
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    epochs = 100
    print("Starting training loop")
    for _ in range(epochs):
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            map, vector = batch

            # Forward pass
            map_hat, vector_hat = model(map)

            # Compute loss
            loss = augmented_loss(map_hat, map, vector_hat, vector)
            loss.backward()

            optimizer.step()

        # Display last batch
        for b in range(map.shape[0]):
            map_hat_disp = detach(map_hat[b])
            map_disp = detach(map[b])
            vector_hat_disp = detach(vector_hat[b])
            vector_disp = detach(vector[b])

            # Quick lambda function to convert to RGB
            to_rgb = lambda x: np.moveaxis(x, 0, -1) * 255

            # Show predicted and actual map
            cv2.imwrite(f"images/{b}_map_hat.jpg", to_rgb(map_hat_disp))
            cv2.imwrite(f"images/{b}_map.jpg", to_rgb(map_disp))

            # Show predicted and actual vector
            vector_hat_disp = np.moveaxis(vector_hat_disp, 0, -1)
            vector_hat_disp = np.argmax(vector_hat_disp, axis=-1)
            vector_hat_disp = vector_hat_disp * (255 // NUM_SEGMENTS)
            cv2.imwrite(f"images/{b}_vector_hat.jpg", (vector_hat_disp).astype(np.uint8))
            vector_disp = np.moveaxis(vector_disp, 0, -1)
            vector_disp = vector_disp * (255 // NUM_SEGMENTS)
            cv2.imwrite(f"images/{b}_vector.jpg", (vector_disp).astype(np.uint8))

    # Create non-decoder model
    model_encoder = AE(
        input_shape=(TILE_SIZE, TILE_SIZE),
        pixel_size=pixel_size,
        in_channels=3,
        vectors=NUM_SEGMENTS,
        decode=False,
    )
    state_dict = {k.replace("module.", ""): v for k, v in model.state_dict().items() if "decode" not in k}
    model_encoder.load_state_dict(state_dict)

    model_scripted = torch.jit.script(model_encoder) # Export to TorchScript
    model_scripted.save(f"models/model_{pixel_size}.pth") # Save
    

# Run the training loop
if __name__ == "__main__":
    pixel_size = int(sys.argv[1])
    train(pixel_size=pixel_size)
