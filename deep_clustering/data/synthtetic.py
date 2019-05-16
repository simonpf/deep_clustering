import matplotlib as mpl
mpl.use("agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.patches import Polygon
import numpy as np
import os
import glob

def create_figure():
    f = plt.figure(frameon = False)
    f.set_size_inches(1, 1)
    ax = plt.Axes(f, [0.0, 0.0, 1.0, 1.0])
    #ax.set_axis_off()
    f.add_axes(ax)

    ax.set_xlim([0, 50])
    ax.set_ylim([0, 50])

    #
    # Circe of random size
    #
    f.patch.set_facecolor("black")
    ax.patch.set_facecolor("black")
    n = np.random.randint(0, 20)
    for i in range(n):
        r = np.random.uniform(5, 10)
        x, y = np.random.uniform(0, 100, size = 2)
        circle = plt.Circle((x, y), r, color = "white")
        ax.add_artist(circle)

    #
    # Random triangle
    #

    n = np.random.randint(0, 20)
    for i in range(n):
        r = np.random.uniform(2, 10)
        x, y = np.random.uniform(0, 100, size = 2)

        v0 = np.array([x, y])
        v1 = v0 + np.array([r, 0])
        v2 = np.array([x + 0.5 * r, y + r * np.sin(60.0 / 180.0 * np.pi)])

        points = np.stack([v0, v1, v2])
        midpoint = (v0 + v1 + v2) / 3.0
        polygon = Polygon(points, color = "white")

        theta = np.random.uniform(0, 360)
        r = Affine2D().rotate_around(midpoint[0], midpoint[1], theta)
        tra = r + ax.transData
        polygon.set_transform(tra)
        ax.add_artist(polygon)

    #
    # Random line
    #

    n = np.random.randint(0, 20)
    for i in range(n):
        l = 2 * np.random.normal() + 10
        x, y = np.random.uniform(0, 100, size = 2)

        theta = np.random.uniform(0, 360)
        v0 = np.array([x, y])
        v1 = v0 + np.array([np.cos(theta)])

        v0 = np.array([x, y])

        theta = np.random.uniform(0, 2 * np.pi)
        v1 = v0 + np.array([l * np.cos(theta), l * np.sin(theta)])

        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], c = "white", lw = 3)
    return f

def create_data(output_path, n):
    for i in range(n):
        imgs = glob.glob(os.path.join(output_path, "img_*.png"))
        ii = len(imgs)
        f = create_figure()
        f.savefig(os.path.join(output_path, "img_{}.png".format(ii)), dpi = 100)

output_path = "/home/simon/src/deep_clustering/data"
