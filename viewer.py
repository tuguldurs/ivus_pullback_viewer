from __future__ import annotations


import argparse


from pydicom import dcmread


def main(args) -> None:
    """Drives the viewer."""
    dcm = dcmread(args.series_name)
    ivus = dcm.pixel_array


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('series_name',
                        help='path to input DICOM Mutiframe-Series')
    args = parser.parse_args()

    main(args)
