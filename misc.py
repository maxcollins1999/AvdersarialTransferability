### Preamble ##########################################################################################################

"""
Misc helper functions.
"""

#######################################################################################################################

### Imports ###########################################################################################################

import numpy as np
import torch
import PIL.Image
from typing import Union, Iterable, Optional, Tuple

#######################################################################################################################

def array_to_PIL(image: Union[np.ndarray, torch.Tensor], channel_first: bool = True) -> PIL.Image:
    """
    :param image: np.ndarray or torch.Tensor
        (c, h, w), (h, w, c), or (h, w) image.
    :param channel_first: bool
        Whether the input is in channel first or channel last format. Note for 2D inputs this argument is ignored.

    Returns a PIL.Image.Image object for the given input image. Note that image values are assumed to be pixel values
    and are clamped between 0 and 255.
    """

    if isinstance(image, torch.Tensor):
        image = np.array(image)
    elif not isinstance(image, np.ndarray):
        raise TypeError("'image' must be a pytorch Tensor or numpy array")

    if image.ndim == 3:
        if channel_first:
            image = np.transpose(image, (1, 2, 0))
        if not image.shape[-1] in [1, 3, 4]:
            raise ValueError(f"Channel dims expected 1, 3, or 4, got {image.shape[-1]}")
    elif image.ndim != 2:
        raise ValueError("'image' expected to be 2 or 3 dimensions, got " f"{image.ndim}")

    image = np.clip(image, 0.0, 255.0)
    image = image.astype(np.uint8)

    return PIL.Image.fromarray(image)

#######################################################################################################################
