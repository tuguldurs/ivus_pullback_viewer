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
from imageio import imread as iio_imread
from imageio import get_writer as iio_get_writer


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
        slider    - Frame slider widget coordinates.
        search_in - Header tag search input textbox widget coordinates.
        console   - 'Console' output textbox widget coordinates.
        header    - Full header display button widget coordinates.
        rmscale   - Scale remover button widget coordinates.
        annotate  - Annotation button widget coordinates.
        save_data - Save button widget coordinates.
    """

    slider     = [0.15, 0.04, 0.60, 0.02]
    search_in  = [0.80, 0.90, 0.25, 0.06]
    console    = [0.04, 0.90, 0.55, 0.06]
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
    def console(coords) -> TextBox:
        """Creates text output box."""
        console = TextBox(
            plt.axes(coords), 
            '', 
            initial='value'
            )
        return console

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
            'Save with GIF',
            hovercolor=self.color
            )
        return save


class PointAnnotator:
    """Creates and logs annotation points."""
    def __init__(self, point):
        self.point = point
        self.xs = list(point.get_xdata())
        self.ys = list(point.get_ydata())
        self.cid = point.figure.canvas.mpl_connect('button_press_event', self)
        with open('annotations.dat', 'w') as handler:
            ...

    def __call__(self, event):
        with open('annotations.dat', 'a') as handler:
            handler.write(f'{event}\n')
        if event.inaxes!=self.point.axes:
            return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.point.set_data(self.xs, self.ys)
        self.point.figure.canvas.draw()


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

        self._cleanup()
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
        self.console = self.widget.console(self.coords.console)
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

    def _cleanup(self) -> None:
        """Cleans any files from previous session."""
        if os.path.isdir(self.output_path):
            rmtree(self.output_path)
            log.info('old output directory removed.')
        files = ['annotations.dat', 'header.txt']
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
                log.info(f'old file cleaned - {file}.')

    def _output_dir(self) -> None:
        """Initializes output directory."""
        os.mkdir(self.output_path)
        log.info('output directory created.')

    def base_figure(self, bot=0.15, left=0.05) -> tuple:
        """Generates base figure, axis, and image objects."""
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=bot, left=left)
        img = ax.imshow(self.start_frame_blank)
        ax.axis('off')
        return fig, ax, img

    def update_frame(self, val) -> None:
        """Updates image with current frame from slider."""
        current_idx = int(round(self.slider.val))
        self.img.set_data(self.pb.video[current_idx])
        self.current_idx = current_idx

    def submit(self, tag_name) -> None:
        """Searches for a tag by name in header."""
        non_valid_str = 'not a valid tag name'
        try:
            header_line = self.pb.dcm.data_element(tag_name)
        except KeyError:
            header_line = f':{non_valid_str}'
        if not header_line:
            value = non_valid_str
        else:
            value = f'{header_line}'.split(':')[-1].strip()
        self.console.set_val(value)

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
        point, = self.ax.plot([], [], 
            linestyle="none", 
            marker="o", 
            color="r")
        _ = PointAnnotator(point)

    def _gif_frame_idx(self) -> np.ndarray:
        """Fetches list of frames indices for gif."""
        start_idx = self.current_idx - 5 if self.current_idx >= 5 else 0
        if self.current_idx <= self.end_frame - 5:
            end_idx = self.current_idx + 5
        else:
            end_idx = self.end_frame
        n_idx = end_idx - start_idx + 1
        return np.arange(n_idx) + start_idx

    def _save_frame(self, idx: int, context: str) -> None:
        """Saves image based on input index and context."""
        frame = self.pb.video[idx]
        frame = self._remove_scale(frame)
        fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
        savename = f'{self.output_path}/{context}_{idx:04}.png'
        plt.savefig(savename)
        plt.close()
        log.info(f'{context} frame saved in {savename}.')

    def _make_gif(self, idxs: np.ndarray) -> None:
        """Creates gif based on base indices."""
        inverse_idxs = idxs[::-1][1:]
        looping_idxs = np.append(idxs, inverse_idxs)
        savename = f'{self.output_path}/anim_{self.current_idx:04}.gif'
        with iio_get_writer(savename, mode='I') as gif_writer:
            for i, idx in enumerate(looping_idxs):
                fname = f'{self.output_path}/gif_{idx:04}.png'
                img = iio_imread(fname)
                gif_writer.append_data(img)
                self.console.set_val(f'creating gif ... {i+1} / {len(looping_idxs)}')
        log.info(f'gif animation saved in {savename}.')

    def saver(self, event) -> None:
        """Saves annotated data and associated gif."""
        self._save_frame(self.current_idx, 'raw')
        gif_idxs = self._gif_frame_idx()
        for idx in gif_idxs:
            self._save_frame(idx, 'gif')
            self.console.set_val(f'saving nearby frames ... {idx}')
        self._make_gif(gif_idxs)
        self.console.set_val('done.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    MultiFrameViewer(args)
