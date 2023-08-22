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
from model import AE, TILE_SIZE

import cv2
import numpy as np
import glob

batch_size = 16

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
        vectors=1,  # Edge detection
        decode=True,
    )
    model = nn.DataParallel(model).to("cuda")

    # Get city tifs
    cities = [
        "austin",
        "houston0",
        "houston1",
    ]
    maps = []
    for city in cities:
        maps += glob.glob(f"scratch/map/{city}.tif")

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

            # Upsample the clusters to match the original coverage
            x_hat = nn.Upsample(size=(TILE_SIZE, TILE_SIZE))(x_hat)

            # Compute the CEL between the clusters and the original coverage
            x_hat = torch.squeeze(x_hat, dim=1)
            edge = self.mse_loss(x_hat, x)

            return 0.1*mse + edge
    augmented_loss = AugmentedLoss()

    # Iterate through the maps and coverages, n times
    n = 3
    for _ in range(n):
        for map in maps:
            # Create a dataset object
            dataset = LandCoverageDataset(
                map,
                tile_size=TILE_SIZE,
                scale=1,
            )

            # Create a dataloader object
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training loop
            epochs = 5
            for _ in range(epochs):
                for batch in tqdm(dataloader):
                    optimizer.zero_grad()
                    coverage, map = batch

                    # Move to gpu
                    coverage = coverage.to("cuda")
                    map = map.to("cuda")

                    # Forward pass
                    coverage_hat, map_hat = model(map)

                    # Compute loss
                    loss = augmented_loss(map_hat, map, coverage_hat, coverage)
                    loss.backward()

                    optimizer.step()

                # Display last batch
                for b in range(coverage.shape[0]):
                    coverage_disp = detach(coverage[b])
                    coverage_hat_disp = detach(coverage_hat[b])
                    map_hat_disp = detach(map_hat[b])
                    map_disp = detach(map[b])
                    
                    # Argmax coverage hat
                    coverage_hat_disp = np.argmax(coverage_hat_disp, axis=0)

                    # Show predicted and actual map
                    map_hat_disp = np.moveaxis(map_hat_disp, 0, -1)
                    cv2.imwrite(f"images/{b}_map_hat.jpg", (map_hat_disp*255).astype(np.uint8))
                    map_disp = np.moveaxis(map_disp, 0, -1)
                    cv2.imwrite(f"images/{b}_map.jpg", (map_disp*255).astype(np.uint8))

                    # Show predicted and actual coverage
                    cv2.imwrite(f"images/{b}_coverage_hat.jpg", coverage_hat_disp*255)
                    cv2.imwrite(f"images/{b}_coverage.jpg", coverage_disp*255)

        torch.save(model.state_dict(), f"models/model_{pixel_size}.pth")
    

# Run the training loop
if __name__ == "__main__":
    pixel_size = int(sys.argv[1])
    train(pixel_size=pixel_size)
