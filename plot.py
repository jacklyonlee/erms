import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


def plot_attribution(x, mask, filename, threshold=90):
    c = np.linalg.norm(mask, axis=1)
    c = (c > np.percentile(c, threshold)).astype(int)
    fig = plt.figure(figsize=(15, 15))
    ax = plt.gca(projection=Axes3D.name)
    ax.scatter(
        x[:, 2],
        x[:, 0],
        x[:, 1],
        s=100,
        c=c,
        cmap=ListedColormap(["white", "red"]),
    )
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.grid(False)
    plt.savefig(
        f"{filename}.png",
        bbox_inches="tight",
    )
    plt.close()
