# Torch imports
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from math import floor

# Kmeans imports
from sklearn.cluster import KMeans
from tqdm import tqdm

from osgeo import gdal

NUM_SEGMENTS = 16

class LandCoverageDataset(Dataset):
    def __init__(self, map_path, tile_size=512, scale=30):

        # Get map raster
        self.map_raster = gdal.Open(map_path)

        # Tile size and scale
        self.tile_size = tile_size
        self.scale = scale

        # Get the number of tiles
        self.num_tiles = self.get_num_tiles(self.map_raster)

        # Read map
        print("Loading map array")
        self.map = np.array(self.map_raster.ReadAsArray(), dtype="uint8")

        # Calculate kmeans clusters per tile
        print("Calculating kmeans clusters")
        self.kmeans_clusters = self.kmeans(self.map, n_clusters=NUM_SEGMENTS)
            
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
            floor(size[0] * scale_ratio / self.tile_size),
            floor(size[1] * scale_ratio / self.tile_size),
        )

        return num_tiles

    # Get a tiles of size tile_size at zoom level scale
    def get_tile(self, raster, arr, idx_og):

        # Get num tiles
        num_tiles = self.get_num_tiles(raster)

        # Get xy indices of tile
        idx = np.unravel_index(idx_og, num_tiles)

        # Get scale ratio
        scale_ratio = self.get_scale_ratio(raster)

        # Get the tile size
        tile_size_trans = self.tile_size / scale_ratio

        # Get the transformed indices
        idx_trans = (idx[0] * tile_size_trans, idx[1] * tile_size_trans)

        # Get the tile
        tile = arr[:,
                   floor(idx_trans[0]) : floor(idx_trans[0] + tile_size_trans),
                   floor(idx_trans[1]) : floor(idx_trans[1] + tile_size_trans)]

        # Convert to torch for gpu
        tile = torch.tensor(tile).cuda().byte()

        # Add batch dimension
        tile = torch.unsqueeze(tile, 0)

        # Resize the tile
        tile = torch.nn.Upsample(size=(self.tile_size, self.tile_size), mode="nearest")(tile)

        return torch.squeeze(tile)

    # Torch Kmeans implementation
    def kmeans(self, map, n_clusters=8):

        # How many pixels to group
        res = 8

        # Pad map to res
        channel_pad = (0, 0)
        height_pad = (0, res - self.map.shape[1] % res)
        width_pad = (0, res - self.map.shape[2] % res)
        map_padded = np.pad(map, (channel_pad, height_pad, width_pad), mode="constant")

        # Reshape map to (h/res*w/res, c*res*res)
        vectors = map_padded.reshape(
            map_padded.shape[0],
            map_padded.shape[1] // res,
            res,
            map_padded.shape[2] // res,
            res
        )
        vectors = vectors.transpose(1, 3, 0, 2, 4)
        vectors = vectors.reshape(vectors.shape[0] * vectors.shape[1], vectors.shape[2] * res * res)

        # Fit kmeans on randum subset of vectors
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        subset = vectors[np.random.choice(vectors.shape[0], 100000, replace=False)]
        kmeans = kmeans.fit(subset)

        # Get cluster labels
        clusters = kmeans.predict(vectors).astype("uint8")

        # Reshape into image (n, c, h, w)
        clusters = clusters.reshape(1, 1, map_padded.shape[-2] // res, map_padded.shape[-1] // res)

        # Convert to torch for gpu
        clusters = torch.tensor(clusters).cuda()

        # Rescale to original size
        clusters = torch.nn.Upsample(size=(map.shape[-2], map.shape[-1]), mode="nearest")(clusters)

        # Get rid of only batch dimension
        clusters = clusters[0]

        return clusters.cpu().numpy()

    def __len__(self):
        y_tiles, x_tiles = self.num_tiles
        return y_tiles * x_tiles

    def __getitem__(self, idx):

        # Get the map tiles
        map_tile = self.get_tile(
            self.map_raster,
            self.map,
            idx
        )
        
        # Get coverage tiles generated from kmeans
        coverage_tile = self.get_tile(
            self.map_raster,
            self.kmeans_clusters,
            idx
        )

        # Scale map tile to float [0, 1]
        map_tile = map_tile.float() / 255

        return coverage_tile.long(), map_tile.float()

# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        "scratch/map/austin.tif",
        tile_size=512,
        scale=1,
    )

    # Display a set of tiles
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create figure
    fig = plt.figure(figsize=(20, 20))

    # Create grid
    gs = gridspec.GridSpec(4, 4)

    # Plot map
    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.imshow(dataset.map.transpose(1, 2, 0))

    # Plot coverage
    ax = fig.add_subplot(gs[2:4, 0:2])
    ax.imshow(dataset.kmeans_clusters.transpose(1, 2, 0))
    
    # Plot random selection of map tiles alongside coverage tiles
    for i in range(4):
        # Get random index
        idx = np.random.randint(0, len(dataset))

        # Get map tile
        map_tile = dataset[idx][1].cpu().numpy()

        # Get coverage tile
        coverage_tile = dataset[idx][0].cpu().numpy()

        # Plot map tile
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(map_tile.transpose(1, 2, 0))

        # Plot coverage tile
        ax = fig.add_subplot(gs[i, 3])
        ax.imshow(coverage_tile)
        
    # Save figure
    fig.savefig("images/land_coverage.png")
