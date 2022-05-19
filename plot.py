import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D


def _create_fig(lim=0.8):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca(projection=Axes3D.name)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    return fig, ax


def _save_fig(filename):
    plt.savefig(
        f"{filename}.png",
        bbox_inches="tight",
    )
    plt.close()


def plot_pc(x, filename, mask=None):
    _, ax = _create_fig()
    ax.scatter(
        x[:, 2],
        x[:, 0],
        x[:, 1],
        s=30,
        c="white" if mask is None else np.linalg.norm(mask, axis=1),
        cmap=None
        if mask is None
        else LinearSegmentedColormap.from_list("heat", ["white", "red"]),
    )
    _save_fig(filename)


def animate_pc(xs, filename):
    fig, ax = _create_fig()

    def update(x):
        ax.clear()
        return ax.scatter(
            x[:, 2],
            x[:, 0],
            x[:, 1],
            s=30,
            c="white",
        )

    ani = FuncAnimation(fig, update, frames=xs, repeat=True)
    ani.save(f"{filename}.gif", writer=PillowWriter(fps=25))
