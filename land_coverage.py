# Torch imports
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from math import floor

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
    def __init__(self, coverage_path, map_path, tile_size=512, scale=30):

        # Get coverage raster
        self.coverage_raster = gdal.Open(coverage_path)

        # Get map raster
        self.map_raster = gdal.Open(map_path)

        # Tile size and scale
        self.tile_size = tile_size
        self.scale = scale

        # Get the number of tiles
        self.num_tiles = self.get_num_tiles(self.coverage_raster)
        assert self.num_tiles == self.get_num_tiles(self.map_raster)

        # Read map
        print("Loading map")
        self.map = np.array(self.map_raster.ReadAsArray(), dtype="uint8")
        self.map = np.swapaxes(self.map, 0, -1)

        # Average bands
        # print("Averaging map bands")
        # self.map = self.map.mean(axis=0)

        # Read coverage
        print("Loading coverage")
        self.coverage = np.array(self.coverage_raster.ReadAsArray(), dtype="uint8")

        # Map coverage tiles to integers
        self.coverage = np.vectorize(nlcd_map.get)(self.coverage)

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
    def get_num_tiles(self, raster):
            
        # Get raster size
        size = self.get_size(raster)

        # Get scale ratio
        scale_ratio = self.get_scale_ratio(raster)

        # Get the number of tiles
        # Scale is used to get the number of tiles at a given zoom level
        num_tiles = (
            floor(size[0]*scale_ratio // self.tile_size),
            floor(size[1]*scale_ratio // self.tile_size),
        )

        return num_tiles

    # Get a tiles of size tile_size at zoom level scale
    def get_tile(self, raster, arr, idx):

        idx = np.unravel_index(idx, self.num_tiles)         

        # Get scale ratio
        scale_ratio = self.get_scale_ratio(raster)

        # Get the tile size
        tile_size = int(self.tile_size / scale_ratio)

        # Get the transformed indices
        idx = int(idx[0] // scale_ratio), int(idx[1] // scale_ratio)

        # Get the tile
        tile = arr[idx[0] : idx[0] + tile_size,
                   idx[1] : idx[1] + tile_size]
        
        # Adjust axes for torch
        if tile.ndim == 3:
            tile = np.swapaxes(tile, 0, -1)
        elif tile.ndim == 2:
            tile = np.expand_dims(tile, axis=0)

        # Convert to torch for gpu
        tile = torch.tensor(tile).cuda().byte()

        # Add batch dimension
        tile = torch.unsqueeze(tile, 0)

        # Resize the tile
        tile = torch.nn.Upsample(size=(self.tile_size, self.tile_size), mode="nearest")(tile)

        return torch.squeeze(tile)

    def __len__(self):
        y_tiles, x_tiles = self.num_tiles
        return y_tiles * x_tiles

    def __getitem__(self, idx):

        # Get the tiles
        map_tiles = self.get_tile(self.map_raster, self.map, idx)
        coverage_tiles = self.get_tile(self.coverage_raster, self.coverage, idx)

        # Scale map tiles
        map_tiles = map_tiles.float() / 255

        return coverage_tiles.long(), map_tiles.float()


# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        "data/nlcd_2019_land_cover_l48_20210604.img",
        "data/map.tif",
        tile_size=512,
        scale=30,
    )
    print(dataset[0])