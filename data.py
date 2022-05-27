import glob
import os

import h5py
import numpy as np
from torch.utils.data import Dataset

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")):
        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system("wget {}  --no-check-certificate; unzip {}".format(www, zipfile))
        os.system("mv {} {}".format(zipfile[:-4], DATA_DIR))
        os.system("rm %s" % (zipfile))


def load_data(partition: str) -> tuple[np.ndarray, np.ndarray]:
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_data = []
    all_label = []
    for h5_name in glob.glob(
        os.path.join(
            DATA_DIR, "modelnet40_ply_hdf5_2048", "ply_data_%s*.h5" % partition
        )
    ):
        f = h5py.File(h5_name, "r")
        data = f["data"][:].astype("float32")
        label = f["label"][:].astype("int64")
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(Dataset):

    classes = (
        "airplane",
        "bathtub",
        "bed",
        "bench",
        "bookshelf",
        "bottle",
        "bowl",
        "car",
        "chair",
        "cone",
        "cup",
        "curtain",
        "desk",
        "door",
        "dresser",
        "flower_pot",
        "glass_box",
        "guitar",
        "keyboard",
        "lamp",
        "laptop",
        "mantel",
        "monitor",
        "night_stand",
        "person",
        "piano",
        "plant",
        "radio",
        "range_hood",
        "sink",
        "sofa",
        "stairs",
        "stool",
        "table",
        "tent",
        "toilet",
        "tv_stand",
        "vase",
        "wardrobe",
        "xbox",
    )

    def __init__(self, num_points: int, partition: str = "test"):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        pointcloud = self.data[idx][: self.num_points]
        label = self.label[idx]
        return pointcloud, label

    def __len__(self) -> int:
        return self.data.shape[0]
