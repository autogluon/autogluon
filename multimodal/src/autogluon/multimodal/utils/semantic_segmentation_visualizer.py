import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Visualizer:
    def __init__(self, img_path: str):
        """
        Parameters
        ----------
            img_path
                File path of the image.
        """
        image = Image.open(img_path)
        plt.imshow(image)

    def draw_image(self, pred: np.array, output_path: str = None):
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

        plt.axis("off")
        if output_path:
            plt.savefig(output_path)
        plt.show()

    def reset_image(self, img_path):
        """
        Parameters
        ----------
            img_path: same as in __init__
        """
        image = Image.open(img_path)
        plt.imshow(image)
