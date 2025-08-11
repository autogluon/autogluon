# Copyright (c) Facebook, Inc. and its affiliates.
# Disclaimer: Special thanks to the Detectron2 developers
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/visualizer.py!
# We use part of its provided, open-source functionalities.
import collections
import colorsys
import logging
import re
from enum import Enum, unique
from typing import List

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from .colormap import random_color
from .misc import merge_spans

logger = logging.getLogger(__name__)


_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


@unique
class ColorMode(Enum):
    """
    Enum of different color modes to use for instance visualizations.
    """

    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


def _create_text_labels(classes: List[str], scores: List[float]):
    """
    Create the label tags for visualization
    Parameters
    ----------
    classes (list[str]): class names for all the detected instances
    scores (list[float]); detection confidence scores for all the detected instances

    Returns
    -------
    labels (list[str]): label tags for visualization
    """
    labels = None
    if classes is not None:
        labels = classes

    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class VisImage:
    def __init__(self, img, scale=1.0):
        """
        Parameters
        ----------
            img (ndarray): an RGB image of shape (H, W, 3) in range [0, 255].
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Parameters
        ----------
            Same as in :meth:`__init__()`.

        Returns
        -------
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        """
        Parameters
        ----------
            img: same as in __init__
        """
        img = img.astype("uint8")
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

    def save(self, filepath):
        """
        Parameters
        ----------
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns
        -------
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class ObjectDetectionVisualizer:
    """
    Visualizer that draws data about detection on images.

    It contains methods like `draw_{text,box}`
    that draw primitive objects to images, as well as high-level wrappers like
    `draw_{instance_predictions}` that draw composite data in some pre-defined style.

    Note that the exact visualization style for the high-level wrappers are subject to change.
    Style such as color, opacity, label contents, visibility of labels, or even the visibility
    of objects themselves (e.g. when the object is too small) may change according
    to different heuristics, as long as the results still look visually reasonable.

    To obtain a consistent style, you can implement custom drawing functions with the
    abovementioned primitive methods instead.  This class does not intend to satisfy
    everyone's preference on drawing styles.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_path, scale=1.0, instance_mode=ColorMode.IMAGE):
        """
        Parameters
        ----------
        img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
            the height and width of the image respectively. C is the number of
            color channels. The image is required to be in RGB format since that
            is a requirement of the Matplotlib library. The image is also expected
            to be in the range [0, 255].
        metadata (Metadata): dataset metadata (e.g. class names and colors)
        instance_mode (ColorMode): defines one of the pre-defined style for drawing
            instances on an image.
        """
        try:
            import cv2
        except:
            raise ImportError("No module named: cv2. Please install cv2 by 'pip install opencv-python'")

        img_rgb = cv2.imread(img_path)
        img_rgb = img_rgb[:, :, ::-1]
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(np.sqrt(self.output.height * self.output.width) // 90, 10 // scale)
        self._instance_mode = instance_mode

    @staticmethod
    def process_predictions(predictions: pd.DataFrame, conf_threshold: float = 0.4):
        """
        Process the classes, box coordinates and confidence scores of the predictions in the image

        Parameters
        ----------
        predictions (pd.DataFrame): the output of object detection with 2 attributes:
            "image": containing paths to the source image
            "bboxes": containing detection results for the images with the following format
                {"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}
        conf_threshold (float): detection confidence threshold to display instances

        Returns
        -------
        boxes: XYXY format of bounding boxes shape = (N, 4)
        scores: detection confidence scores, shape = (N, )
        classes: detection classes, shape = (N, )
        """
        boxes, scores, classes = [], [], []
        instances = predictions["bboxes"]
        for instance in instances:
            s = instance["score"]
            if s >= conf_threshold:
                box = instance["bbox"]
                c = instance["class"]
                boxes.append(box)
                scores.append(s)
                classes.append(c)
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)
        assert len(boxes) == len(scores) == len(classes), (
            "Expected boxes, scores and classes to have the same length, but got len(boxes): {}, len(scores) = {}, len(classes) = {}".format(
                len(boxes), len(scores), len(classes)
            )
        )
        if len(boxes) == 0:
            return None, None, None
        return boxes, scores, classes

    def draw_instance_predictions(self, predictions: pd.DataFrame, conf_threshold: float = 0.4):
        """
        Draw instance-level prediction results on an image.

        Parameters
        ----------
        predictions (pd.DataFrame): the output of object detection for that image, with 2 attributes:
            "image": containing paths to the source image
            "bboxes": containing detection results for the images with the following format
                {"class": <predicted_class_name>, "bbox": [x1, y1, x2, y2], "score": <confidence_score>}
        conf_threshold (float): detection confidence threshold to display instances

        Returns
        -------
        output (VisImage): image object with visualizations.
        """
        boxes, scores, classes = self.process_predictions(predictions, conf_threshold=conf_threshold)
        labels = _create_text_labels(classes, scores)
        colors = None

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy() if predictions.has("pred_masks") else None
                )
            )

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
        )
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        assigned_colors=None,
    ):
        """
        Draw the visualizations
        Parameters
        ----------
        boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
            or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
            or a :class:`RotatedBoxes`,
            or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
            for the N objects in a single image,
        labels (list[str]): the text to be displayed for each instance.
        assigned_colors (list[matplotlib.colors]): a list of colors, where each color
            corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
            for full list of formats that the colors are accepted in.
        Returns
        -------
            output (VisImage): image object with visualizations.
        """
        num_instances = 0
        if boxes is not None:
            num_instances = len(boxes)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale or y1 - y0 < 40 * self.output.scale:
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        return self.output

    """
    Primitive drawing functions:
    """

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Parameters
        ----------
        text (str): class label
        position (tuple): a tuple of the x and y coordinates to place text on image.
        font_size (int, optional): font of the text. If not provided, a font size
            proportional to the image width is calculated and used.
        color: color of the text. Refer to `matplotlib.colors` for full list
            of formats that are accepted.
        horizontal_alignment (str): see `matplotlib.text.Text`
        rotation: rotation angle in degrees CCW

        Returns
        -------
        output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
        """
        Parameters
        ----------
        box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
            are the coordinates of the image's top left corner. x1 and y1 are the
            coordinates of the image's bottom right corner.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        edge_color: color of the outline of the box. Refer to `matplotlib.colors`
            for full list of formats that are accepted.
        line_style (string): the string to use to create the outline of the boxes.

        Returns
        -------
        output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    """
    Internal methods:
    """

    def _create_grayscale_image(self, mask=None):
        """
        Create a grayscale version of the original image.
        The colors in masked area, if given, will be kept.
        """
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color, brightness_factor):
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Parameters
        ----------
        color: color of the polygon. Refer to `matplotlib.colors` for a full list of
            formats that are accepted.
        brightness_factor (float): a value in [-1.0, 1.0] range. A lightness factor of
            0 will correspond to no change, a factor in [-1.0, 0) range will result in
            a darker color and a factor in (0, 1.0] range will result in a lighter color.

        Returns
        -------
        modified_color (tuple[double]): a tuple containing the RGB values of the
            modified color. Each value in the tuple is in the [0.0, 1.0] range.
        """
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color


class SemanticSegmentationVisualizer:
    """
    Visualize images and predicted semantic segmentation masks.
    """

    def plot_image(self, img_path: str):
        """
        Parameters
        ----------
            img_path
                File path of the image.
        """
        image = Image.open(img_path)
        plt.imshow(image)

    def plot_mask(self, pred: np.array, output_path: str = None):
        """
        Parameters
        ----------
            pred
                np.array of the mask prediction
            output_path
                The path to save the mask image.
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        class_ids = np.unique(pred)
        for class_id in class_ids:
            if class_id == 0:  # background
                continue
            show_mask(pred == class_id, plt.gca())

        if output_path:
            plt.savefig(output_path)
        plt.show()


class NERVisualizer:
    """An NER visualizer that renders NER prediction as a string of HTML
    inline to any Python class Jupyter notebooks.
    """

    def __init__(self, pred, sent, seed):
        self.pred = pred
        self.sent = sent
        self.colors = {}
        self.spans = merge_spans(sent, pred, for_visualizer=True)
        self.spans = collections.OrderedDict(sorted(self.spans.items()))
        self.rng = np.random.RandomState(seed)

    @staticmethod
    def escape_html(text: str) -> str:
        """Replace <, >, &, " with their HTML encoded representation. Intended to
        prevent HTML errors in rendered displaCy markup.
        text (str): The original text.
        RETURNS (str): Equivalent text to be safely used within HTML.
        """
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        return text

    def html_template(self, text, label, color):
        """
        Generate an HTML template for the given text and its label.

        Parameters
        ----------
        text
            The text to be highlighted.
        label
            The predicted label for the given text.
        color
            The background color of the mark tag.
        """
        text = '<mark style="background-color:{}; color:white; border-radius: .6em .6em; padding: .1em;">{} \
         <b style="background-color:white; color:black; font-size:x-small; border-radius: 0.5em .5em; padding: .0em;">{} </b> \
         </mark>'.format(color, self.escape_html(text), self.escape_html(label))
        return text

    def _repr_html_(self):
        entities = []
        new_sent = ""
        last = 0
        for key, value in self.spans.items():
            entity_group = value[-1]
            if re.match("B-", entity_group, re.IGNORECASE) or re.match("I-", entity_group, re.IGNORECASE):
                entity_group = entity_group[2:]
            if entity_group not in self.colors:
                self.colors.update({entity_group: "#%06X" % self.rng.randint(0, 0xFFFFFF)})
            start = key
            new_sent += self.sent[last:start]
            last = end = value[0]
            entity_text = self.html_template(self.sent[start:end], entity_group, color=self.colors[entity_group])
            new_sent += entity_text
        new_sent += self.sent[last:]

        return new_sent


def visualize_ner(sentence, prediction, seed=0):
    """
    Visualize the prediction of NER.

    Parameters
    ----------
    sentence
        The input sentence.
    prediction
        The NER prediction for the sentence.
    seed
        The seed for colorpicker.

    Returns
    -------
    An NER html visualizer.
    """
    visualizer = NERVisualizer(prediction, sentence, seed)
    return visualizer
