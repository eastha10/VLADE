# Derived from AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456.
# Apache-2.0 / inherited MIT: src/vlade/models/teachers/AVA_LICENSE.txt.
from __future__ import annotations
from typing import Any, List, Tuple, Union
import numpy as np
from PIL import Image

OPENVLA_IMAGE_SIZE = 224


def resize_image_for_policy(
    img: np.ndarray, resize_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    import tensorflow as tf

    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    import tensorflow as tf

    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    import tensorflow as tf

    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        check_image_format(image)

        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        pil_image = Image.fromarray(image).convert("RGB")

        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images
