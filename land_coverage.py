# Torch imports
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from math import floor, sqrt

# Kmeans imports
from sklearn.cluster import KMeans
from tqdm import tqdm

from osgeo import gdal

CLUSTERS_MIN = 2
CLUSTERS_MAX = 8

class LandCoverageDataset(Dataset):
    def __init__(self, map_path, tile_size=512, scale=30):

        # Get map raster
        self.map_raster = gdal.Open(map_path)

        # Tile size and scale
        self.tile_size = tile_size
        self.scale = scale

        # Get num tiles
        self.num_tiles = self.get_num_tiles(self.map_raster)

        # Read map
        print("Loading map array")
        self.map = np.array(self.map_raster.ReadAsArray(), dtype="uint8")

        # Crop map to tile size
        self.map = self.map[
            :,
            : self.map.shape[1] - self.map.shape[1] % tile_size,
            : self.map.shape[2] - self.map.shape[2] % tile_size,
        ]

        # Create tiles to calculate kmeans clusters
        tiles = self.map.reshape(
            self.map.shape[0],
            self.map.shape[1] // tile_size,
            tile_size,
            self.map.shape[2] // tile_size,
            tile_size,
        )
        tiles = tiles.transpose(1, 3, 0, 2, 4)
        tiles = tiles.reshape(
            self.map.shape[1] // tile_size * self.map.shape[2] // tile_size,
            self.map.shape[0],
            tile_size,
            tile_size,
        )

        # Calculate "best" kmeans clusters per tile
        print("Calculating kmeans clusters")
        kmeans_tiles = []
        for tile in tqdm(tiles):
            kmeans_tiles.append(self.best_kmeans(tile, min_clusters=CLUSTERS_MIN, n_clusters=CLUSTERS_MAX))

        # Reshape kmeans clusters to match map
        kmeans_tiles = np.stack(kmeans_tiles, axis=0)
        self.kmeans_clusters = kmeans_tiles.reshape(
            self.map.shape[1] // tile_size,
            self.map.shape[2] // tile_size,
            tile_size,
            tile_size,
        )
        self.kmeans_clusters = self.kmeans_clusters.transpose(0, 2, 1, 3)
        self.kmeans_clusters = self.kmeans_clusters.reshape(self.map.shape[1:])

        # Get edge map of kmeans clusters per tile
        print("Calculating edge map")
        self.edge_map = []
        for tile in tqdm(kmeans_tiles):
            self.edge_map.append(self.get_edge_map(tile))
        
        # Reshape edge map to match map
        self.edge_map = np.stack(self.edge_map, axis=0)
        self.edge_map = self.edge_map.reshape(
            self.map.shape[1] // tile_size,
            self.map.shape[2] // tile_size,
            tile_size,
            tile_size,
        )
        self.edge_map = self.edge_map.transpose(0, 2, 1, 3)
        self.edge_map = self.edge_map.reshape(1, *self.map.shape[1:])

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

    # Get the edge map of a map
    def get_edge_map(self, map):

        # Get the edge map
        edge_map = np.zeros(map.shape, dtype="uint8")
        for i in range(map.shape[0]):
            edge_map[i] = cv2.Canny(map[i], 0, np.max(map[i]))

        # Average the edge map channels
        edge_map = np.mean(edge_map, axis=0, keepdims=True)
        edge_map[edge_map > 0] = 1

        # Display tile with edge map
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(map.transpose(1, 2, 0))
        # ax[1].imshow(edge_map[0])
        # plt.show()

        return edge_map.astype("bool")

    # Best kmeans calculator
    def best_kmeans(self, tile, min_clusters=2, n_clusters=8):

        # Get the best kmeans clusters for the tile
        min_improvement = 1e8
        best_inertia = np.inf
        best_clusters = None
        for n in range(min_clusters, n_clusters + 1):
            clusters, inertia = self.kmeans(tile, res=4, n_clusters=n)
            if inertia < best_inertia and best_inertia - inertia > min_improvement:
                best_inertia = inertia
                best_clusters = clusters
            else:
                break
            
        # Display the best clusters with tile
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(tile.transpose(1, 2, 0))
        # ax[1].imshow(best_clusters[0])
        # plt.show()
        
        return best_clusters

    # CPU Kmeans implementation
    def kmeans(self, map, res=1, n_clusters=8):

        assert res > 0, "res must be greater than 0"
        assert map.shape[1] % res == 0, "vectors must be divisible by res"
        assert map.shape[2] % res == 0, "vectors must be divisible by res"
        
        # Reshape map to (h/res*w/res, c*res*res)
        res_shape = map.shape[1] // res, map.shape[2] // res
        vectors = map.reshape(
            map.shape[0],
            map.shape[1] // res,
            res,
            map.shape[2] // res,
            res
        )
        vectors = vectors.transpose(1, 3, 0, 2, 4)
        vectors = vectors.reshape(vectors.shape[0] * vectors.shape[1], vectors.shape[2] * res * res)

        # Fit kmeans on randum subset of vectors
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
        kmeans = kmeans.fit(vectors)

        # Get clusters and mean inertia
        clusters = kmeans.labels_.astype("uint8")
        inertia = kmeans.inertia_

        # Reshape into image (n, c, h, w)
        clusters = clusters.reshape(1, 1, *res_shape)

        # Convert to torch for gpu
        clusters = torch.tensor(clusters).cuda()

        # Rescale to original size
        clusters = torch.nn.Upsample(size=(map.shape[-2], map.shape[-1]), mode="nearest")(clusters)

        # Get rid of only batch dimension
        clusters = clusters[0]

        return clusters.cpu().numpy(), inertia

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
        edge_tile = self.get_tile(
            self.map_raster,
            self.edge_map,
            idx
        )

        # Scale map tile to float [0, 1]
        map_tile = map_tile.float() / 255

        return edge_tile.long(), map_tile.float()

# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        "scratch/map/test.tif",
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
    # ax = fig.add_subplot(gs[0:2, 0:2])
    # ax.imshow(dataset.map.transpose(1, 2, 0))

    # Plot coverage
    # ax = fig.add_subplot(gs[2:4, 0:2])
    # ax.imshow(dataset.edge_map[0])
    
    # Plot random selection of map tiles alongside coverage tiles
    for i in range(16):
        # Get random index
        idx = np.random.randint(0, len(dataset))

        # Get map tile
        map_tile = dataset[idx][1].cpu().numpy()

        # Get coverage tile
        edge_tile = dataset[idx][0].cpu().numpy()

        # Plot map tile
        ax = fig.add_subplot(gs[i // 4 * 2, i % 4])
        ax.imshow(map_tile.transpose(1, 2, 0))

        # Plot coverage tile
        ax = fig.add_subplot(gs[i // 4 * 2 + 1, i % 4])
        ax.imshow(edge_tile[0])
        
    # Save figure
    fig.savefig("images/land_coverage.png")
