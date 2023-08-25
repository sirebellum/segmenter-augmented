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

from model import TILE_SIZE, NUM_SEGMENTS, AE

class LandCoverageDataset(Dataset):
    def __init__(self, map_path, tile_size=512, scale=30, pixel_size=1):

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

        # Calculate "best" kmeans clusters per tile
        print("Calculating kmeans clusters")
        self.kmeans_clusters, _ = self.kmeans(
            self.map,
            res=pixel_size,
            n_clusters=NUM_SEGMENTS,
        )

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
        tile = torch.tensor(tile).byte()

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
    def best_kmeans(self, tile, min_clusters=2, n_clusters=8, pixel_size=1):

        # Get the best kmeans clusters for the tile
        improvement_factor = 1.1
        best_inertia = np.inf
        best_clusters = None
        for n in range(min_clusters, n_clusters + 1):
            clusters, inertia = self.kmeans(tile, res=pixel_size, n_clusters=n)
            if best_inertia/inertia > improvement_factor:
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
        sample_size = 42690 if vectors.shape[0] > 42690 else vectors.shape[0]
        subset = np.random.choice(vectors.shape[0], sample_size, replace=False)
        kmeans = kmeans.fit(vectors[subset])

        # Get clusters and mean inertia
        clusters = kmeans.predict(vectors).astype("uint8")
        inertia = kmeans.inertia_

        # Reshape into image (n, c, h, w)
        clusters = clusters.reshape(1, 1, *res_shape)

        # Convert to torch
        clusters = torch.tensor(clusters)

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
        cluster_tile = self.get_tile(
            self.map_raster,
            self.kmeans_clusters,
            idx
        )

        # Scale map tile to float [0, 1]
        map_tile = map_tile.float() / 255

        return cluster_tile.long(), map_tile.float()

# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        "scratch/map/test.tif",
        tile_size=512,
        scale=1,
        pixel_size=4,
    )

    model = torch.jit.load("models/model_4.pth").cuda().eval()

    # Display a set of tiles
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Create figure
    fig = plt.figure(figsize=(10, 20))
    gs = gridspec.GridSpec(2, 1)

    # Plot map
    ax = fig.add_subplot(gs[0, 0])
    map_array = dataset.map
    ax.imshow(map_array.transpose(1, 2, 0))

    # Calculate and plot coverage
    ax = fig.add_subplot(gs[1, 0])
    channel_padding = (0, 0)
    height_padding = (0, TILE_SIZE - map_array.shape[1] % TILE_SIZE)
    width_padding = (0, TILE_SIZE - map_array.shape[2] % TILE_SIZE)
    map_padded = np.pad(map_array, (channel_padding, height_padding, width_padding), mode="constant")
    map_tiles = map_padded.reshape(
        map_padded.shape[0],
        map_padded.shape[1] // TILE_SIZE,
        TILE_SIZE,
        map_padded.shape[2] // TILE_SIZE,
        TILE_SIZE,
    )
    map_tiles = map_tiles.transpose(1, 3, 0, 2, 4)
    map_tiles = map_tiles.reshape(
        map_tiles.shape[0] * map_tiles.shape[1],
        map_tiles.shape[2],
        TILE_SIZE,
        TILE_SIZE,
    )
    map_tiles = torch.tensor(map_tiles).cuda().float() / 255
    coverage_map = model(map_tiles)
    coverage_map = torch.exp(coverage_map)
    coverage_map = torch.argmax(coverage_map, dim=1).type(torch.uint8)
    coverage_map = coverage_map.cpu().numpy()
    coverage_map = coverage_map.reshape(
        map_padded.shape[1] // TILE_SIZE,
        map_padded.shape[2] // TILE_SIZE,
        coverage_map.shape[1],
        coverage_map.shape[2],
    )
    coverage_map = coverage_map.transpose(0, 2, 1, 3)
    coverage_map = coverage_map.reshape(
        1, 1,
        coverage_map.shape[0] * coverage_map.shape[1],
        coverage_map.shape[2] * coverage_map.shape[3],
    )
    coverage_map = torch.tensor(coverage_map).cuda()
    coverage_map = torch.nn.Upsample(size=(map_array.shape[1], map_array.shape[2]), mode="nearest")(coverage_map)
    coverage_map = coverage_map[
        0,
        :,
        : map_array.shape[1],
        : map_array.shape[2],
    ].cpu().numpy()
    ax.imshow(coverage_map[0])

    # Save figure
    fig.savefig("images/edge_map.png")

    # Create figure
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(4, 2)
    
    # Plot random selection of map tiles alongside coverage tiles
    used_idx = []
    for i in range(4):
        # Get random index
        idx = np.random.randint(0, len(dataset))
        while idx in used_idx:
            idx = np.random.randint(0, len(dataset))

        # Get map tile
        map_tile = dataset[idx][1].cpu().numpy()

        # Get coverage tile
        coverage_map = dataset[idx][0].cpu().numpy()

        # Plot map tile
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(map_tile.transpose(1, 2, 0))

        # Plot coverage tile
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(coverage_map)
        
    # Save figure
    fig.savefig("images/land_coverage.png")
