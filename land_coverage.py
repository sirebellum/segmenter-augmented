# Torch imports
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

from osgeo import gdal

# Map nlcd land cover classes to integers
nlcd_map = {
    0: 0,
    11: 1,
    12: 2,
    21: 3,
    22: 4,
    23: 5,
    24: 6,
    31: 7,
    41: 8,
    42: 9,
    43: 10,
    51: 11,
    52: 12,
    71: 13,
    72: 14,
    73: 15,
    74: 16,
    81: 17,
    82: 18,
    90: 19,
    95: 20,
}
NUM_CLASSES = 20 + 1


class LandCoverageDataset(Dataset):
    def __init__(self, coverage_path, map_path, tile_size=512):

        # Get coverage raster
        self.coverage_raster = gdal.Open(coverage_path)

        # Get map raster
        self.map_raster = gdal.Open(map_path)

        # Tile size ratio for memory
        self.tile_ratio = 4
        self.tile_size = tile_size

        # Turn the rasters into numpy arrays of tiles
        self.coverage_tiles, self.map_tiles = self.tile_data(
            self.map_raster, self.coverage_raster, tile_size//self.tile_ratio
        )

        self.num_tiles = self.coverage_tiles.shape[0]

    # Turn the raster into a grid of tiles
    def tile_data(self, map, coverage, tile_size):

        # Get numpy arrays from rasters
        map_array = np.array(map.ReadAsArray(), dtype="uint8")
        coverage_array = np.array(coverage.ReadAsArray(), dtype="uint8")

        # Use coverage map sizes
        map_width, map_height = coverage_array.shape[0]//2, coverage_array.shape[1]//2

        # Resize map array
        map_array = np.moveaxis(map_array, 0, -1)
        map_array = cv2.resize(map_array, (map_height, map_width))

        # Resize to something manageable
        coverage_array = cv2.resize(coverage_array, (map_height, map_width), interpolation=cv2.INTER_NEAREST)

        # Get the number of tiles in each dimension
        num_tiles_x = map_width // tile_size
        num_tiles_y = map_height // tile_size

        # Get the number of tiles
        num_tiles = num_tiles_x * num_tiles_y

        # Create empty arrays to hold the tiles
        map_tiles = np.zeros((num_tiles, tile_size, tile_size, 3), dtype="uint8")
        coverage_tiles = np.zeros((num_tiles, tile_size, tile_size), dtype="uint8")

        # Loop through the tiles
        for x in range(num_tiles_x):
            for y in range(num_tiles_y):
                # Get map tile
                map_tiles[y*num_tiles_x + x] = map_array[
                    x * tile_size : (x + 1) * tile_size,
                    y * tile_size : (y + 1) * tile_size,
                ]
                # Get coverage tile
                coverage_tiles[y*num_tiles_x + x] = coverage_array[
                    x * tile_size : (x + 1) * tile_size,
                    y * tile_size : (y + 1) * tile_size,
                ]
        
        # Delete arrays
        del map_array
        del coverage_array

        # Get rid of tiles containing pixels with no class
        truth_map = coverage_tiles != 0
        map_tiles = map_tiles[np.all(truth_map, axis=(1, 2))]
        coverage_tiles = coverage_tiles[np.all(truth_map, axis=(1, 2))]

        # Map the coverage tiles to integers
        coverage_tiles = np.vectorize(nlcd_map.get)(coverage_tiles)

        # Move the channels to the front for pytorch
        map_tiles = np.moveaxis(map_tiles, -1, 1)

        return coverage_tiles, map_tiles

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):

        # Get the tiles
        map_tiles = self.map_tiles[idx]
        coverage_tiles = self.coverage_tiles[idx]

        # Upscale the tiles to tile_size
        map_tiles = cv2.resize(map_tiles, (self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)
        coverage_tiles = cv2.resize(coverage_tiles, (self.tile_size, self.tile_size), interpolation=cv2.INTER_NEAREST)

        # Scale map tiles
        map_tiles = map_tiles.astype("float32") / 255

        return torch.tensor(coverage_tiles).long(), torch.tensor(map_tiles).float()


# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        "data/nlcd_2019_land_cover_l48_20210604.img", "data/map.tif", tile_size=512
    )

    # Construct map and display