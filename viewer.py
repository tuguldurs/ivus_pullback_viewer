from __future__ import annotations

import os
import sys
import argparse
import logging
import subprocess
from shutil import rmtree
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
        annotate - Annotation button widget coordinates.
        save_data - Save button widget coordinates.
    """

    slider     = [0.23, 0.02, 0.56, 0.04]
    search_in  = [0.20, 0.90, 0.25, 0.08]
    search_out = [0.50, 0.90, 0.40, 0.08]
    header     = [0.80, 0.80, 0.20, 0.04]
    rmscale    = [0.80, 0.70, 0.20, 0.04]
    annotate   = [0.80, 0.60, 0.20, 0.04]
    save_data  = [0.80, 0.50, 0.20, 0.04]


class WidgetCreator:
    """Creates various widget objects."""

    def __init__(self) -> None:
        self.color = 'green'
    
    def slider(self, coords, start_frame, end_frame) -> Slider:
        """Creates slider object."""
        slider = Slider(
            plt.axes(coords),
            'Frame #: ', 
            start_frame, end_frame,
            valinit=start_frame,
            valstep=1,
            color=self.color
            )
        return slider

    @staticmethod
    def search_in(coords) -> TextBox:
        """Creates tag search input box."""
        search_in = TextBox(
            plt.axes(coords),
            'Search Tag: ', 
            initial='tag'
            )
        return search_in

    @staticmethod
    def search_out(coords) -> TextBox:
        """Creates tag search output box."""
        search_out = TextBox(
            plt.axes(coords), 
            '', 
            initial='tag value'
            )
        return search_out

    def header(self, coords) -> Button:
        """Creates header viewer button."""
        header = Button(
            plt.axes(coords),
            'Show Full Header', 
            hovercolor=self.color
            )
        return header

    def rmscale(self, coords) -> Button:
        """Creates scale remover button."""
        rmscale = Button(
            plt.axes(coords),
            'Remove Scale', 
            hovercolor=self.color
            )
        return rmscale

    def annotate(self, coords) -> Button:
        """Creates annotation button."""
        annotate = Button(
            plt.axes(coords),
            'Annotate',
            hovercolor=self.color
            )
        return annotate

    def save_data(self, coords) -> Button:
        """Creates save button."""
        save = Button(
            plt.axes(coords),
            'Save',
            hovercolor=self.color
            )
        return save


class LineBuilder:
    """Creates and logs annotation points."""
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        with open('annotations.dat', 'w') as handler:
            ...

    def __call__(self, event):
        with open('annotations.dat', 'a') as handler:
            handler.write(f'{event}\n')
        if event.inaxes!=self.line.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


class MultiFrameViewer:
    """Main viewer object."""

    def __init__(self, args) -> None:

        # Loads pullback data.
        self.pb = PullBack(args.series_name)

        # Initializes parameters and sets up output dir.
        self.start_frame = 0
        self.end_frame = len(self.pb.video) - 1
        self.start_frame_blank = [[1,0],[0,1]]
        self.output_path = 'OUTPUT'
        self._output_dir()

        # Sets up base figure.
        self.fig, self.ax, self.img = self.base_figure()

        # Creates widgets.
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
        self.annotate = self.widget.annotate(self.coords.annotate)
        self.save_data = self.widget.save_data(self.coords.save_data)

        # Assigns actions.
        self.slider.on_changed(self.update_frame)
        self.search_in.on_submit(self.submit)
        self.header.on_clicked(self.open_header)
        self.rmscale.on_clicked(self.remove_scale)
        self.annotate.on_clicked(self.annotator)
        self.save_data.on_clicked(self.saver)

        plt.show()

    def _output_dir(self) -> None:
        """Initializes output directory."""
        if os.path.isdir(self.output_path):
            log.info('old output directory removed.')
            rmtree(self.output_path)
        log.info('output directory created.')
        os.mkdir(self.output_path)

    def base_figure(self, bot=0.15, left=0.05) -> tuple:
        """Generates base figure, axis, and image objects."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=bot, left=left)
        img = ax.imshow(self.start_frame_blank)
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

    @staticmethod
    def _remove_scale(frame) -> np.ndarray:
        """Removes scale marks from input raw frame."""
        red_channel = np.dot(frame[... , :3] , [1, 0, 0])
        avg_signal = np.dot(frame[... , :3] , [1, 1, 1]) / 3
        diff = red_channel - avg_signal
        frame[diff > 0] = np.array([0,0,0])
        return frame

    def remove_scale(self, event) -> None:
        """Removes scale marks from current frame."""
        raw_frame = self.pb.video[self.current_idx]
        fixed_frame = self._remove_scale(raw_frame)
        self.img.set_data(fixed_frame)

    def annotator(self, event) -> None:
        """Applies point annotations."""
        line, = self.ax.plot([], [], 
            linestyle="none", 
            marker="o", 
            color="r")
        linebuilder = LineBuilder(line)

    def saver(self, event) -> None:
        """Saves annotated data and associated gif."""
        raw_frame = self.pb.video[self.current_idx]
        fixed_frame = self._remove_scale(raw_frame)
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.imshow(fixed_frame, cmap='gray')
        ax.axis('off')
        plt.savefig(f'{self.output_path}/raw_{self.current_idx:04}.png')
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    MultiFrameViewer(args)
