from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from isobenefit import land_map
from rasterio import transform

TEMP_DIR = Path("temp/")
TEMP_DIR.mkdir(parents=False, exist_ok=True)


def test_plot(land: land_map.Land):
    """ """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=True, sharex=True, sharey=True)
    fig.suptitle(f"Iteration {land.iters}")
    axes[0][0].imshow(land.state_arr, origin="lower")
    axes[0][0].set_title("land state")
    axes[0][1].imshow(land.green_itx_arr, origin="lower")
    axes[0][1].set_title("green periphery")
    axes[1][0].imshow(land.green_acc_arr, origin="lower")
    axes[1][0].set_title("green access")
    axes[1][1].imshow(land.density_arr, origin="lower")
    axes[1][1].set_title("density")
    plt.tight_layout()
    plt.savefig(TEMP_DIR / f"{land.iters}.png", dpi=200)


def test_land_map():
    """ """
    x_span = 6000
    y_span = 4000
    granularity_m = 50
    max_distance_m = 500
    # y is rows
    extents_arr = np.full((int(y_span / granularity_m), int(x_span / granularity_m)), 0, dtype=np.int_)
    # snip out corner of extents for testing out of bounds
    for x_idx, y_idx in np.ndindex(extents_arr.shape):
        dist = np.hypot(x_idx * granularity_m, y_idx * granularity_m)
        if dist <= 200:
            extents_arr[x_idx, y_idx] = -1
    # snips
    extents_arr[:, 0] = -1
    extents_arr[1, :] = -1
    land = land_map.Land(
        granularity_m=granularity_m,
        max_distance_m=max_distance_m,
        # prepare transform - expects w, s, e, n, cell width, cell height
        extents_transform=transform.from_bounds(0, 0, x_span, y_span, extents_arr.shape[1], extents_arr.shape[0]),
        extents_arr=extents_arr,
        centre_seeds=[(int(x_span / 3), int(y_span / 2))],
        min_green_km2=1,
        random_seed=20,
    )
    test_plot(land)
    iters = 500
    for _ in range(iters):
        land.iterate()
        print(land.iters)
        if land.iters < 10 or land.iters % 50 == 0:
            test_plot(land)
    test_plot(land)
    print("here")


def test_recurse_gobble():
    """ """
    test_arr = np.full((100, 100), 0, dtype=np.int_)
    # start in middle
    enough_extents = land_map.continuous_state_extents(test_arr, 50, 50, 0, 200 * 200, 250, 50)
    assert enough_extents
    # start in corner
    enough_extents = land_map.continuous_state_extents(test_arr, 0, 0, 0, 200 * 200, 250, 50)
    assert enough_extents
    # reduce distance - should return False since start node is not included
    enough_extents = land_map.continuous_state_extents(test_arr, 0, 0, 0, 200 * 200, 200, 50)
    assert enough_extents is False


if __name__ == "__main__":
    test_land_map()
    # test_recurse_gobble()
