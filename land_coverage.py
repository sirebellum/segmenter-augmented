# Torch imports
import torch
import torchvision
import numpy as np
import cv2
from torch.utils.data import Dataset
from math import floor

# Kmeans imports
from sklearn.cluster import KMeans
from tqdm import tqdm

from osgeo import gdal

from model import TILE_SIZE, NUM_SEGMENTS, AE

class LandCoverageDataset(Dataset):
    def __init__(self, tile_size=512, scale=30):

        # Set up metdata
        self.tile_size = tile_size
        self.scale = scale

        # Get map raster
        self.map_raster = gdal.Open("scratch/map/map.tif")
        self.map = np.array(self.map_raster.ReadAsArray(), dtype="uint8")

        # Load coverage data
        self.coverage_raster = gdal.Open("scratch/coverage/coverage.tif")
        self.coverage = np.array(self.coverage_raster.ReadAsArray(), dtype="uint8")

        # Convert coverage array to continuous integers
        self.coverage_ints = np.unique(self.coverage)
        self.coverage_ints = np.arange(len(self.coverage_ints))
        for i, c in enumerate(np.unique(self.coverage)):
            self.coverage[self.coverage == c] = self.coverage_ints[i]

        # Resize coverage to map size
        self.coverage = cv2.resize(
            self.coverage,
            (self.map.shape[2], self.map.shape[1]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Pad to tile size
        channel_padding = (0, 0)
        height_padding = (0, tile_size - self.map.shape[1] % tile_size)
        width_padding = (0, tile_size - self.map.shape[2] % tile_size)
        height_padding = height_padding if height_padding[1] != 512 else (0, 0)
        width_padding = width_padding if width_padding[1] != 512 else (0, 0)
        self.map = np.pad(self.map, (channel_padding, height_padding, width_padding), mode="constant")
        self.coverage = np.pad(self.coverage, (height_padding, width_padding), mode="constant")

        # Num of tiles
        self.num_tiles = self.get_num_tiles(self.map)

        # Add depth dimension
        self.coverage = np.expand_dims(self.coverage, axis=0)

    # Get raster size
    def get_size(self, raster):
        return raster.RasterYSize, raster.RasterXSize

    # Get the scale ratio
    def get_scale_ratio(self, raster):

        # Get map resolution per pixel
        res = raster.GetGeoTransform()[1]

        # Get ratio of user provided scale to map resolution
        scale_ratio = res / self.scale

        return scale_ratio

    # Get the number of tiles
    def get_num_tiles(self, array):
            
        # Get raster size
        size = array.shape

        assert array.shape[1] % self.tile_size == 0, "tile_size must be divisible by map height"
        assert array.shape[2] % self.tile_size == 0, "tile_size must be divisible by map width"

        # Get the number of tiles
        num_tiles = (
            size[1] // self.tile_size,
            size[2] // self.tile_size,
        )

        return num_tiles

    # Get a tiles of size tile_size at zoom level scale
    def get_tile(self, arr, idx_og):

        # Get num tiles
        num_tiles = self.num_tiles

        # Get xy indices of tile
        idx = np.unravel_index(idx_og, num_tiles)

        # Get the tile
        tile = arr[
            :,
            idx[0] * self.tile_size : idx[0] * self.tile_size + self.tile_size,
            idx[1] * self.tile_size : idx[1] * self.tile_size + self.tile_size
        ]

        return tile

    # Randomly permute the tile
    def permute(self, map_tile):
        
        # Randomly flip the tile
        if np.random.rand() > 0.5:
            map_tile = torch.flip(map_tile, dims=(1,))
        
        # Randomly rotate the tile up to 3 times
        for _ in range(np.random.randint(0, 4)):
            map_tile = torch.rot90(map_tile, k=1, dims=(1, 2))
        
        return map_tile

    def __len__(self):
        return np.prod(self.num_tiles)

    def __getitem__(self, idx):

        # Get the map tiles
        map_tile = self.get_tile(
            self.map,
            idx
        )

        # Convert to torch tensors
        map_tile = torch.tensor(map_tile).float()

        # Randomly permute (flip, rotate, saturation, etc.)
        # map_tile = self.permute(map_tile)

        # Scale map tile to float [0, 1]
        map_tile = map_tile / 255

        # Get the coverage tile
        coverage_tile = self.get_tile(
            self.coverage,
            idx
        )

        # Convert to torch tensors
        coverage_tile = torch.tensor(coverage_tile).long()

        return map_tile.cuda(), coverage_tile.cuda()

# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        tile_size=1024*4,
        scale=1,
    )

    # Display whole map
    # map = dataset.map
    # coverage = dataset.coverage
    # map = np.moveaxis(map, 0, -1)
    # coverage = np.moveaxis(coverage, 0, -1) * (255 / NUM_SEGMENTS)
    # map = cv2.resize(map, (map.shape[1] // 2, map.shape[0] // 2))
    # coverage = cv2.resize(coverage, (coverage.shape[1] // 2, coverage.shape[0] // 2))
    # cv2.imwrite("images/map.jpg", map)
    # cv2.imwrite("images/coverage.jpg", coverage)

    # Display a few tiles
    for i in range(10):
        map_tile, coverage_tile = dataset[i]

        # Convert to numpy array
        map_tile = map_tile.cpu().numpy()
        coverage_tile = coverage_tile.cpu().numpy()

        # Convert to RGB
        map_tile = np.moveaxis(map_tile, 0, -1) * 255

        # Convert to RGB
        coverage_tile = np.moveaxis(coverage_tile, 0, -1) * 255 / NUM_SEGMENTS

        # Display
        cv2.imwrite(f"images/{i}_map_tile.jpg", map_tile)
        cv2.imwrite(f"images/{i}_coverage_tile.jpg", coverage_tile)
    