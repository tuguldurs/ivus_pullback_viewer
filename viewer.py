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
        self.search_in = self._get_search_in()
        self.search_out = self._get_search_out()

    def _get_slider(self) -> list:
        """Creates slider coordinates."""
        return [0.23, 0.02, 0.56, 0.04]

    def _get_search_in(self) -> list:
        """Creates tag search box coordinates."""
        return [0.2, 0.9, 0.25, 0.08]

    def _get_search_out(self) -> list:
        """Creates tag search output box coordinates."""
        return [0.5, 0.9, 0.4, 0.08]


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
        self.search_in = self._get_search_in(self.coords.search_in)
        self.search_in.on_submit(self.submit)
        self.search_out = self._get_search_out(self.coords.search_out)
        plt.show()

    def base_figure(self, bot=0.15, left=0.05) -> tuple:
        """Generates base figure, axis, and image objects."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=bot, left=left)
        img = ax.imshow([[1,0],[0,1]])
        return fig, ax, img

    def _get_slider(self, coords, start_frame, end_frame) -> Slider:
        """Creates slider object."""
        ax_slider = plt.axes(coords)
        slider = Slider(
            ax_slider, 'Frame #: ', start_frame, end_frame,
            valinit=start_frame,
            valstep=1,
            color='green'
            )
        return slider

    def _get_search_in(self, coords) -> TextBox:
        """Creates tag search input box."""
        ax_search_in = plt.axes(coords)
        tag_search_input = TextBox(ax_search_in, 'Search Tag: ', initial='tag')
        return tag_search_input

    def _get_search_out(self, coords) -> TextBox:
        """Creates tag search output box."""
        ax_search_out = plt.axes(coords)
        tag_search_output = TextBox(ax_search_out, '', initial='tag value')
        return tag_search_output

    def update_frame(self, val) -> None:
        """Updates image with current frame from slider."""
        current_idx = int(round(self.slider.val))
        self.img.set_data(self.pb.video[current_idx])

    def submit(self, tag_name) -> None:
        non_valid_str = 'not a valid tag'
        try:
            header_line = self.pb.dcm.data_element(tag_name)
        except KeyError:
            header_line = f':{non_valid_str}'
        if not header_line:
            value = non_valid_str
        else:
            value = f'{header_line}'.split(':')[-1].strip()
        self.search_out.set_val(value)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    Viewer(args)
