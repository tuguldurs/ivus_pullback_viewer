from __future__ import annotations

import os
import sys
import argparse
import logging
import subprocess
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


@dataclass
class WidgetCoords:
    """Sets widget coordinates [x, y, dx, dy].
    
    Attributes:
        slider - Frame slider widget coordinates.
        search_in - Header tag search input textbox widget coordinates.
        search_out - Header tag search output textbox widget coordinates.
        header - Full header display button widget coordinates.
        rmscale - Scale remover button widget coordinates.
    """

    slider     = [0.23, 0.02, 0.56, 0.04]
    search_in  = [0.20, 0.90, 0.25, 0.08]
    search_out = [0.50, 0.90, 0.40, 0.08]
    header     = [0.80, 0.80, 0.20, 0.04]
    rmscale    = [0.80, 0.70, 0.20, 0.04]


class WidgetCreator:
    """Creates various widgets."""
    
    @staticmethod
    def slider(coords, start_frame, end_frame) -> Slider:
        """Creates slider object."""
        ax_slider = plt.axes(coords)
        slider = Slider(
            ax_slider, 'Frame #: ', 
            start_frame, end_frame,
            valinit=start_frame,
            valstep=1,
            color='green'
            )
        return slider

    @staticmethod
    def search_in(coords) -> TextBox:
        """Creates tag search input box."""
        ax_search_in = plt.axes(coords)
        search_in = TextBox(
            ax_search_in, 'Search Tag: ', 
            initial='tag'
            )
        return search_in

    @staticmethod
    def search_out(coords) -> TextBox:
        """Creates tag search output box."""
        ax_search_out = plt.axes(coords)
        search_out = TextBox(
            ax_search_out, '', 
            initial='tag value'
            )
        return search_out

    @staticmethod
    def header(coords) -> Button:
        """Creates header viewer button."""
        ax_header = plt.axes(coords)
        header = Button(
            ax_header, 'Show Full Header', 
            hovercolor='0.975'
            )
        return header

    @staticmethod
    def rmscale(coords) -> Button:
        """Creates scale remover button."""
        ax_rmscale = plt.axes(coords)
        rmscale = Button(
            ax_rmscale, 'Remove Scale', 
            hovercolor='0.975'
            )
        return rmscale


class Viewer:
    """Main viewer object."""

    def __init__(self, args) -> None:
        self.pb = PullBack(args.series_name)
        self.start_frame = 0
        self.end_frame = len(self.pb.video) - 1
        self.fig, self.ax, self.img = self.base_figure()

        # widgets
        self.coords = WidgetCoords()
        self.widget = WidgetCreator()
        self.slider = self.widget.slider(
            self.coords.slider, 
            self.start_frame, 
            self.end_frame
            )
        self.search_in = self.widget.search_in(self.coords.search_in)
        self.search_out = self.widget.search_out(self.coords.search_out)
        self.header = self.widget.header(self.coords.header)
        self.rmscale = self.widget.rmscale(self.coords.rmscale)

        # actions
        self.slider.on_changed(self.update_frame)
        self.search_in.on_submit(self.submit)
        self.header.on_clicked(self.open_header)
        self.rmscale.on_clicked(self.remove_scale)

        plt.show()

    def base_figure(self, bot=0.15, left=0.05) -> tuple:
        """Generates base figure, axis, and image objects."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=bot, left=left)
        img = ax.imshow([[1,0],[0,1]])
        return fig, ax, img

    def update_frame(self, val) -> None:
        """Updates image with current frame from slider."""
        current_idx = int(round(self.slider.val))
        self.img.set_data(self.pb.video[current_idx])
        self.current_idx = current_idx

    def submit(self, tag_name) -> None:
        """Searches for a tag by name in header and 
            outputs results in search_out."""
        non_valid_str = 'not a valid tag name'
        try:
            header_line = self.pb.dcm.data_element(tag_name)
        except KeyError:
            header_line = f':{non_valid_str}'
        if not header_line:
            value = non_valid_str
        else:
            value = f'{header_line}'.split(':')[-1].strip()
        self.search_out.set_val(value)

    def open_header(self, event) -> None:
        """Creates and opens full header as txt file."""
        filename = 'header.txt'
        with open(filename, 'w') as handler:
            for tag in self.pb.dcm.dir():
                handler.write(f'{self.pb.dcm.data_element(tag)}\n')
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

    def remove_scale(self, event) -> None:
        """Removes scale marks from current frame."""
        raw_frame = self.pb.video[self.current_idx]
        red_channel = np.dot(raw_frame[... , :3] , [1, 0, 0])
        avg_signal = np.dot(raw_frame[... , :3] , [1, 1, 1]) / 3
        diff = red_channel - avg_signal
        raw_frame[diff > 0] = np.array([0,0,0])
        self.img.set_data(raw_frame)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    Viewer(args)
