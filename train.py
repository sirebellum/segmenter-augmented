# Imports neded for torch training
import torch
import torch.nn as nn
import torch.optim as optim

# Imports needed for data loading
from torch.utils.data import DataLoader
from land_coverage import LandCoverageDataset, num_classes

# Imports needed for training loop
from tqdm import tqdm

# Import model
from model import AE

import cv2
import numpy as np

pixel_size = 16
tile_enum = {16: 520}
tile_size = tile_enum[pixel_size]
batch_size = 8

# main training function
def train():
    # Create a dataset object
    dataset = LandCoverageDataset(
        "scratch/nlcd_2019_land_cover_l48_20210604.img",
        "scratch/map.tif",
        tile_size=tile_size,
    )

    # Create a dataloader object
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = AE(
        input_shape=(tile_size, tile_size),
        pixel_size=pixel_size,
        in_channels=3,
        decode=True,
        augmented=True,
        vectors=num_classes
    ).to("cuda")

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
            x_hat = nn.Upsample(size=(tile_size, tile_size))(x_hat)

            # Compute the CEL between the clusters and the original coverage
            cel = self.cel_loss(x_hat, x)

            return mse + cel
    augmented_loss = AugmentedLoss()

    # Training loop
    for epoch in range(1000):
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
            coverage_disp = coverage[b].cpu().detach().numpy()
            coverage_hat_disp = coverage_hat[b].cpu().detach().numpy()
            map_hat_disp = map_hat[b].cpu().detach().numpy()
            map_disp = map[b].cpu().detach().numpy()
            
            # Argmax coverage hat
            coverage_hat_disp = np.argmax(coverage_hat_disp, axis=0)

            # Show predicted and actual map
            map_hat_disp = np.moveaxis(map_hat_disp, 0, -1)
            cv2.imwrite(f"images/{b}_map_hat.jpg", (map_hat_disp*255).astype(np.uint8))
            map_disp = np.moveaxis(map_disp, 0, -1)
            cv2.imwrite(f"images/{b}_map.jpg", (map_disp*255).astype(np.uint8))

            # Show predicted and actual coverage
            cv2.imwrite(f"images/{b}_coverage_hat.jpg", coverage_hat_disp*10)
            cv2.imwrite(f"images/{b}_coverage.jpg", coverage_disp*10)

    torch.save(model.state_dict(), f"models/model_{pixel_size}.pth")
    

# Run the training loop
if __name__ == "__main__":
    train()
