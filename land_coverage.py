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
    def __init__(self, map_paths, tile_size=512, scale=30, pixel_size=1):

        # Set up metdata
        self.tile_size = tile_size
        self.scale = scale

        # Iterate through map paths
        self.maps = []
        for map_path in map_paths:

            # Get map raster
            map_raster = gdal.Open(map_path)

            # Read map
            print(f"Loading {map_path}")
            map = np.array(map_raster.ReadAsArray(), dtype="uint8")

            # Pad map to tile size
            channel_padding = (0, 0)
            height_padding = (0, tile_size - map.shape[1] % tile_size)
            width_padding = (0, tile_size - map.shape[2] % tile_size)
            map = np.pad(map, (channel_padding, height_padding, width_padding), mode="constant")

            self.maps.append(map)

        # Concatenate
        self.maps = np.stack(self.maps, axis=0)
        self.num_tiles = self.get_num_tiles(self.maps)

        # # Calculate kmeans for the maps
        # print("Calculating kmeans clusters")
        # self.kmeans_clusters, _ = self.kmeans(
        #     self.maps,
        #     res=pixel_size,
        #     n_clusters=NUM_SEGMENTS,
        # )


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

        assert array.shape[2] % self.tile_size == 0, "tile_size must be divisible by map height"
        assert array.shape[3] % self.tile_size == 0, "tile_size must be divisible by map width"

        # Get the number of tiles
        # Scale is used to get the number of tiles at a given zoom level
        num_tiles = (
            size[0],
            size[2] // self.tile_size,
            size[3] // self.tile_size,
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
            idx[0],
            :,
            idx[1] * self.tile_size : idx[1] * self.tile_size + self.tile_size,
            idx[2] * self.tile_size : idx[2] * self.tile_size + self.tile_size
        ]

        return tile

    # # Get the edge map of a map
    # def get_edge_map(self, map):

    #     # Get the edge map
    #     edge_map = np.zeros(map.shape, dtype="uint8")
    #     for i in range(map.shape[0]):
    #         edge_map[i] = cv2.Canny(map[i], 0, np.max(map[i]))

    #     # Average the edge map channels
    #     edge_map = np.mean(edge_map, axis=0, keepdims=True)
    #     edge_map[edge_map > 0] = 1

    #     # Display tile with edge map
    #     # import matplotlib.pyplot as plt
    #     # fig, ax = plt.subplots(1, 2)
    #     # ax[0].imshow(map.transpose(1, 2, 0))
    #     # ax[1].imshow(edge_map[0])
    #     # plt.show()

    #     return edge_map.astype("bool")

    # # Best kmeans calculator
    # def best_kmeans(self, tile, min_clusters=2, n_clusters=8, pixel_size=1):

    #     # Get the best kmeans clusters for the tile
    #     improvement_factor = 1.1
    #     best_inertia = np.inf
    #     best_clusters = None
    #     for n in range(min_clusters, n_clusters + 1):
    #         clusters, inertia = self.kmeans(tile, res=pixel_size, n_clusters=n)
    #         if best_inertia/inertia > improvement_factor:
    #             best_inertia = inertia
    #             best_clusters = clusters
    #         else:
    #             break
            
    #     # Display the best clusters with tile
    #     # import matplotlib.pyplot as plt
    #     # fig, ax = plt.subplots(1, 2)
    #     # ax[0].imshow(tile.transpose(1, 2, 0))
    #     # ax[1].imshow(best_clusters[0])
    #     # plt.show()
        
    #     return best_clusters

    # # CPU Kmeans implementation
    # def kmeans(self, maps, res=1, n_clusters=8):

    #     assert res > 0, "res must be greater than 0"
    #     assert maps.shape[2] % res == 0, "vectors must be divisible by res"
    #     assert maps.shape[3] % res == 0, "vectors must be divisible by res"
        
    #     # Reshape map to (h/res*w/res, c*res*res)
    #     vectors = maps.reshape(
    #         maps.shape[0],
    #         maps.shape[1],
    #         maps.shape[2] // res,
    #         res,
    #         maps.shape[3] // res,
    #         res
    #     )
    #     vectors = vectors.transpose(0, 2, 4, 1, 3, 5)
    #     vectors = vectors.reshape(
    #         vectors.shape[0] * vectors.shape[1] * vectors.shape[2],
    #         vectors.shape[3] * vectors.shape[4] * vectors.shape[5],
    #     )

    #     # Fit kmeans on randum subset of vectors
    #     kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    #     sample_size = 100000 if vectors.shape[0] > 100000 else vectors.shape[0]
    #     subset = np.random.choice(vectors.shape[0], sample_size, replace=False)
    #     kmeans = kmeans.fit(vectors[subset])

    #     # Get clusters and inertia
    #     batch_size = 100000
    #     n_batches = floor(vectors.shape[0] / batch_size)
    #     clusters = []
    #     for i in tqdm(range(n_batches)):
    #         clusters.append(kmeans.predict(vectors[i * batch_size : (i + 1) * batch_size]).astype("uint8"))
    #     clusters.append(kmeans.predict(vectors[n_batches * batch_size :]).astype("uint8"))
    #     clusters = np.concatenate(clusters)
    #     inertia = kmeans.inertia_

    #     # Reshape into image (n, c, h, w)
    #     clusters = clusters.reshape(
    #         maps.shape[0],
    #         1,
    #         maps.shape[-2] // res,
    #         maps.shape[-1] // res,
    #     )

    #     # Convert to torch
    #     clusters = torch.tensor(clusters)

    #     # Rescale to original size
    #     clusters = torch.nn.Upsample(size=(maps.shape[-2], maps.shape[-1]), mode="nearest")(clusters)

    #     return clusters.cpu().numpy(), inertia

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
            self.maps,
            idx
        )
        
        # Get coverage tiles generated from kmeans
        # cluster_tile = self.get_tile(
        #     self.kmeans_clusters,
        #     idx
        # )

        # Convert to torch tensors
        map_tile = torch.tensor(map_tile).cuda()
        # cluster_tile = torch.tensor(cluster_tile).cuda()

        # Randomly permute (flip, rotate, saturation, etc.)
        map_tile = self.permute(map_tile)

        # Scale map tile to float [0, 1]
        map_tile = map_tile.float() / 255

        return map_tile.float()

# run main stuff
if __name__ == "__main__":
    dataset = LandCoverageDataset(
        ["scratch/map/test.tif"],
        tile_size=512,
        scale=1,
        pixel_size=16,
    )

    model = torch.jit.load("models/model_8.pth").eval()

    