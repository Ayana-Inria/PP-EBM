from typing import List
import os
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

FONT_PATH = os.path.join(os.path.split(
    os.path.abspath(__file__))[0], 'Minitel.ttf')


def make_image_from_bunch(ndarray, padding=2, pad_value=0.0):
    if ndarray.ndim == 4:
        ndarray = np.expand_dims(ndarray, -1)

    if ndarray.ndim == 5 and ndarray.shape[-1] == 1:  # single-channel images
        ndarray = np.concatenate((ndarray, ndarray, ndarray), -1)

    # make the mini-batch of images into a grid
    ymaps, xmaps = ndarray.shape[:2]

    height, width = int(ndarray.shape[2] +
                        padding), int(ndarray.shape[3] + padding)
    num_channels = ndarray.shape[4]
    grid = np.full((height * ymaps + padding, width * xmaps +
                   padding, num_channels), pad_value).astype(np.float32)
    for y in range(ymaps):
        for x in range(xmaps):
            grid[y * height + padding:(y + 1) * height, x *
                 width + padding:(x + 1) * width] = ndarray[y, x]

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    # ndarr = np.clip(grid * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return grid


def add_header(image_array, text: str, pad_value=0.0, draw_value=1.0, padding=2):
    height = 3 * padding + 10 + 1

    num_channels = image_array.shape[2]

    header_arr = np.full(
        (height, image_array.shape[1], num_channels), pad_value).astype(np.float32)

    header_arr[-padding, padding:-padding] = draw_value

    int_draw_value = int(draw_value * 255)

    img = Image.fromarray(np.uint8(header_arr * 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 8)
    draw.text((padding, padding), text, (int_draw_value,
              int_draw_value, int_draw_value), font=font)

    header_arr = np.array(img) / 255

    return np.concatenate((header_arr, image_array), axis=0), height


def _legend(texts, padding, num_channels, legend_width, header_size, pad_value, draw_value):
    width = 2 * padding + 10 + 1

    all_legends = []
    unit_width = (legend_width - header_size) // len(texts)
    for t in texts[::-1]:
        legend_arr = np.full((width, unit_width, num_channels),
                             pad_value).astype(np.float32)

        int_draw_value = int(draw_value * 255)

        img = Image.fromarray(np.uint8(legend_arr * 255))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT_PATH, 8)
        draw.text((padding, padding), t, (int_draw_value,
                  int_draw_value, int_draw_value), font=font, align='center')
        legend_arr = np.array(img) / 255
        all_legends.append(legend_arr)

    all_legends = np.concatenate(all_legends, axis=1)
    return all_legends, width, unit_width


def add_left_legend(image_array, texts: List[str], padding=2, pad_value=0.0, draw_value=1.0, header_size=0):
    num_channels = image_array.shape[2]
    all_legends, width, unit_width = _legend(texts, padding, num_channels, image_array.shape[0],
                                             header_size, pad_value, draw_value)

    all_legends = np.rot90(all_legends, 1)
    filler = np.full(
        (image_array.shape[0] - unit_width * len(texts), width, num_channels), pad_value)
    all_legends = np.concatenate([filler, all_legends], axis=0)
    right_fill = np.full(
        (all_legends.shape[0], 2 * padding, num_channels), pad_value)
    right_fill[:, -padding] = draw_value
    all_legends = np.concatenate([all_legends, right_fill], axis=1)

    return np.concatenate((all_legends, image_array), axis=1), all_legends.shape[1]


def add_top_legend(image_array, texts: List[str], padding=2, pad_value=0.0, draw_value=1.0):
    num_channels = image_array.shape[2]
    all_legends, width, unit_width = _legend(texts[::-1], padding, num_channels, image_array.shape[1], 0, pad_value,
                                             draw_value)

    filler = np.full(
        (width, image_array.shape[1] - unit_width * len(texts), num_channels), pad_value)
    all_legends = np.concatenate([filler, all_legends], axis=1)
    bot_fille = np.full(
        (2 * padding, all_legends.shape[1], num_channels), pad_value)
    bot_fille[-padding, :] = draw_value
    all_legends = np.concatenate([all_legends, bot_fille], axis=0)

    return np.concatenate((all_legends, image_array), axis=0), all_legends.shape[0]
