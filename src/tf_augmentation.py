import tensorflow as tf
import math
import numpy as np
from typing import List, Tuple

AUTOTUNE = tf.data.AUTOTUNE


"https://medium.com/@fanzongshaoxing/adjust-local-brightness-for-image-augmentation-8111c001059b"

_SQRT2PI = math.sqrt(2 * math.pi)


def _pdf(x, mu, sigma):
    return tf.exp(-tf.square(x - mu) / (2 * sigma * sigma)) / (sigma * _SQRT2PI)


def generate_blob(x_value_rate: tf.Tensor, centers: List[Tuple[tf.Tensor, tf.Tensor]], dev) -> tf.Tensor:
    center_prob = _pdf(0.0, 0.0, dev)
    N, M = x_value_rate.shape
    for u, v in centers:
        iv, jv = tf.meshgrid(tf.range(N, dtype=tf.float32), tf.range(M, dtype=tf.float32), indexing="ij")
        di, dj = iv - u, jv - v
        d = tf.sqrt(di * di + dj * dj)
        x_value_rate = x_value_rate + _pdf(d, 0.0, dev) / center_prob
    return x_value_rate


"https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow"


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_blur(img, sigma):
    blur = _gaussian_kernel(3, sigma, 3, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], "SAME")
    return img[0]


def random_mask(image: tf.Tensor, mask: tf.Tensor, new_seed, cnt: int = 1):
    N, M, _ = image.shape
    blob = tf.zeros((N, M))
    for I in range(cnt):
        i, j, r2 = (
            tf.random.stateless_uniform(
                shape=[],
                seed=new_seed[I * 3 + t, :],
                minval=(0, 0, 30)[t],
                maxval=(N, M, min(M, N) // (2 * cnt))[t],
                dtype=image.dtype,
            )
            for t in range(3)
        )
        blob = generate_blob(blob, [[i, j]], r2)
    blob = tf.repeat(tf.reshape(blob, [N, M, 1]), 3, -1)
    return mask * blob + image * (1.0 - blob)


BRIGHTNESS_DELTA = (-150, 50)
CONTRAST_RANGE = (0.2, 1.7)
SATURATION_RANGE = (0.2, 1.6)
COLOR_SHIFT = 30.0


def random_blur(image: tf.Tensor, new_seed):
    sigma = tf.random.stateless_uniform(shape=[], seed=new_seed, minval=0.5, maxval=2, dtype=tf.float32)
    return apply_blur(image, sigma)


def random_blur_mask(image: tf.Tensor, new_seed):
    mask = random_blur(image, new_seed[3, :])
    return random_mask(image, mask, new_seed[:3, :])


def random_brightness(image: tf.Tensor, new_seed):
    delta = tf.random.stateless_uniform([], minval=BRIGHTNESS_DELTA[0], maxval=BRIGHTNESS_DELTA[1], dtype=tf.float32, seed=new_seed)
    return tf.image.adjust_brightness(image, delta)


def random_brightness_mask(image: tf.Tensor, new_seed):
    mask = random_brightness(image, new_seed[3, :])
    return random_mask(image, mask, new_seed[:3, :])


def random_contrast(image: tf.Tensor, new_seed):
    return tf.image.stateless_random_contrast(image, lower=CONTRAST_RANGE[0], upper=CONTRAST_RANGE[1], seed=new_seed)


def random_contrast_mask(image: tf.Tensor, new_seed):
    mask = random_contrast(image, new_seed[3, :])
    return random_mask(image, mask, new_seed[:3, :])


def random_saturation(image: tf.Tensor, new_seed):
    return tf.image.stateless_random_saturation(
        image, lower=SATURATION_RANGE[0], upper=SATURATION_RANGE[1], seed=new_seed
    )


def random_saturation_mask(image: tf.Tensor, new_seed):
    mask = random_saturation(image, new_seed[3, :])
    return random_mask(image, mask, new_seed[:3, :])


def random_color(image: tf.Tensor, new_seed):
    return image[:, :] + tf.random.stateless_uniform(
        shape=[3], minval=-COLOR_SHIFT, maxval=COLOR_SHIFT, dtype=image.dtype, seed=new_seed
    )


def random_color_mask(image: tf.Tensor, new_seed):
    mask = random_color(image, new_seed[3, :])
    return random_mask(image, mask, new_seed[:3, :])


def random_rotate(image: tf.Tensor, new_seed):
    return tf.image.rot90(
        image, tf.random.stateless_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32, seed=new_seed)
    )


def _augment(image_label, seed):
    image, label = image_label

    new_seed = tf.random.experimental.stateless_split(seed, num=24)

    image = random_color(image, new_seed[20, :])
    image = random_brightness(image, new_seed[21, :])
    image = random_contrast(image, new_seed[22, :])

    image = random_color_mask(image, new_seed[0:4])
    image = random_brightness_mask(image, new_seed[4:8])
    image = random_contrast_mask(image, new_seed[8:12])
    image = random_saturation_mask(image, new_seed[12:16])
    image = random_blur_mask(image, new_seed[16:20])

    image = tf.clip_by_value(image, 0, 255)

    image = random_rotate(image, new_seed[23, :])

    return image, label


_counter = tf.data.experimental.Counter()


def ds_augment(ds: tf.data.Dataset) -> tf.data.Dataset:
    return tf.data.Dataset.zip((ds, (_counter, _counter))).map(_augment, num_parallel_calls=AUTOTUNE)


__all__ = ["ds_augment"]
