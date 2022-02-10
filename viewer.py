from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from pydicom import dcmread


log  = logging.getLogger(__name__)
fmt = '%(asctime)s ~ %(name)14s ~ %(levelname)8s ::: %(message)s'
lvl = logging.INFO
logging.basicConfig(level=lvl, format=fmt)


@dataclass
class PullBack:
    """Reads input DICOM series instance.

    Attributes:
        series_name - Path to DICOM series instance.
        dcm - pydicom.dataset.Filedataset object.
        video - numpy array of individual frames.
    """

    series_name: str

    def __post_init__(self) -> None:
        self.dcm = dcmread(self.series_name)
        self.video = self.dcm.pixel_array
        self.n_frames = len(self.video)
        log.info(f'DICOM data read from {self.series_name}')
        log.info(f'number of frames = {self.n_frames}')


class WidgetCoords:
    """Generates widget coordinates."""

    def __init__(self) -> None:
        self.slider = self._get_slider()

    def _get_slider(self) -> list:
        """Creates slider coordinates."""
        return [0.23, 0.02, 0.56, 0.04]


class Viewer:
    """Driver object."""

    def __init__(self, args) -> None:
        self.pb = PullBack(args.series_name)
        self.coords = WidgetCoords()
        self.start_frame = 0
        self.end_frame = len(self.pb.video) - 1
        self.fig, self.ax, self.img = self.base_figure()
        self.slider = self._get_slider(self.coords.slider, self.start_frame, self.end_frame)
        self.slider.on_changed(self.update_frame)
        plt.show()

    def base_figure(self, bot=0.15, left=0.05) -> tuple:
        """Generates base figure, axis, and image objects."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=bot, left=left)
        img = ax.imshow([[1,0],[0,1]])
        return fig, ax, img

    def _get_slider(self, coords, start_frame, end_frame):
        """Creates slider object."""
        ax_slider = plt.axes(coords)
        slider = Slider(
            ax_slider, 'Frame #: ', start_frame, end_frame,
            valinit=start_frame,
            valstep=1,
            color='green'
            )
        return slider

    def update_frame(self, val) -> None:
        """Updates image with current frame from slider."""
        current_idx = int(round(self.slider.val))
        self.img.set_data(self.pb.video[current_idx])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    Viewer(args)
