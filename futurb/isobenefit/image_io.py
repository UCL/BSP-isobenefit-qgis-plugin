from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib import cm
from PIL import Image


def import_2Darray_from_image(filepath: Path) -> npt.NDArray[np.float_]:
    """ """
    img = Image.open(filepath)
    img_data: list[Any] = img.getdata()  # type: ignore
    data = np.array(img_data).reshape(img.size[1], img.size[0], -1).mean(axis=2)
    data_rescaled = (data - data.min()) / (data.max() - data.min())
    return data_rescaled


def plot_image_from_2Darray(
    normalized_data_array: npt.NDArray[np.float_],
    color_map: Any = cm.gist_earth,  # type: ignore
) -> None:
    """ """
    data_mapped = np.uint8(255 * color_map(normalized_data_array))
    img = Image.fromarray(data_mapped)  # type: ignore
    img.show()


def save_image_from_2Darray(canvas: npt.NDArray[np.float_], filepath: Path, format: str = "png") -> None:
    """ """
    data_mapped = np.uint8(255 * canvas)
    img = Image.fromarray(data_mapped)  # type: ignore
    img.save(filepath, format=format)
