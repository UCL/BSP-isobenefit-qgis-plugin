from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from isobenefit import land_map
from rasterio import transform

TEMP_DIR = Path("temp/")
TEMP_DIR.mkdir(parents=False, exist_ok=True)


def test_plot(land: land_map.Land, iter: int):
    """ """
    plt.tight_layout()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), squeeze=True, sharex=True, sharey=True)
    axes[0][0].imshow(land.state_arr)
    axes[0][1].imshow(land.green_itx_arr)
    axes[1][0].imshow(land.cent_acc_arr)
    axes[1][1].imshow(land.green_acc_arr)
    plt.savefig(TEMP_DIR / f"{iter}.png", dpi=200)


def test_land_map():
    """ """
    span = 10000
    granularity_m = 40
    walk_dist_m = 1000
    cells = int(span / granularity_m)
    extents_arr = np.full((cells, cells), 0, dtype=np.int_)
    # snip out corner of extents for testing out of bounds
    for x_idx, y_idx in np.ndindex(extents_arr.shape):
        dist = np.hypot(x_idx * granularity_m, y_idx * granularity_m)
        if dist <= 1000:
            extents_arr[x_idx, y_idx] = -1
    land = land_map.Land(
        granularity_m=granularity_m,
        walk_dist_m=walk_dist_m,
        # prepare transform - expects w, s, e, n, cell width, cell height
        extents_transform=transform.from_bounds(0, 0, span, span, extents_arr.shape[0], extents_arr.shape[1]),
        extents_arr=extents_arr,
        centre_seeds=[(5000, 5000)],
    )
    iters = 200
    for iter in range(iters):
        print(iter)
        land.iter_land_isobenefit()
        if iter % 10 == 0:
            test_plot(land, iter)
    test_plot(land, iters)
    print("here")


if __name__ == "__main__":
    test_land_map()
