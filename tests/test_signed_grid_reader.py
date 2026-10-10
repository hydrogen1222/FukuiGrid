"""Read/write regression tests for signed periodic volumetric grids.

From repository root:
    python3 -m unittest discover -s tests -v

These tests verify file integrity only. They do not validate conceptual DFT
physical approximations, electrode models or Fukui reaction predictions.
"""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from FukuiGrid import process_file, read_file_info, write_fukui_file


def header(shape):
    return (
        "Synthetic CHGCAR for signed reader regression\n"
        "1.0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "H\n"
        "1\n"
        "Direct\n"
        "0.0 0.0 0.0\n"
        "\n"
        f"{shape[0]} {shape[1]} {shape[2]}\n"
    )


class TestSignedVolumetricReader(unittest.TestCase):
    def round_trip(self, shape, numbers):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "CHGCAR"
            output = Path(directory) / "FUKUI.vasp"
            array = np.asarray(numbers, dtype=float)
            self.assertEqual(array.size, math.prod(shape))
            source.write_text(header(shape), encoding="utf-8")
            write_fukui_file(str(source), 1, array, str(output))
            atoms, grid, points, _, _, _, skip, max_rows = read_file_info(str(output))
            self.assertEqual(atoms, 1)
            self.assertEqual(grid, list(shape))
            self.assertEqual(points, array.size)
            actual = process_file(str(output), skip, max_rows, points)
            self.assertEqual(actual.size, array.size)
            np.testing.assert_allclose(actual, array, rtol=1e-11, atol=1e-11)
            self.assertEqual(int(np.count_nonzero(actual < 0)),
                             int(np.count_nonzero(array < 0)))
            self.assertEqual(int(np.count_nonzero(actual == 0)),
                             int(np.count_nonzero(array == 0)))

    def test_late_negative_values_after_inference_window(self):
        # The old pandas.read_fwf(colspecs='infer') missed 400 minus signs
        # when its initial width-inference window had no negative samples.
        numbers = np.array([0.112345678901] * 600
                           + [-0.198123456789] * 400)
        self.round_trip((10, 10, 10), numbers)

    def test_zero_negative_and_short_last_row(self):
        self.round_trip((2, 2, 2),
                        [0.125, 0, -0.25, 0.5, 0, 1, -0.75, 0.375])

    def test_all_zeros(self):
        self.round_trip((2, 2, 1), [0, 0, 0, 0])

    def test_sign_preserved_after_two_repeated_io_steps(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "CHGCAR"
            first = Path(directory) / "STEP1"
            second = Path(directory) / "STEP2"
            array = np.asarray([0.25] * 600 + [-0.15] * 400)
            source.write_text(header((10, 10, 10)), encoding="utf-8")
            write_fukui_file(str(source), 1, array, str(first))
            _, _, points, _, _, _, skip, rows = read_file_info(str(first))
            once = process_file(str(first), skip, rows, points)
            self.assertEqual(int(np.count_nonzero(once < 0)), 400)
            write_fukui_file(str(first), 1, once * 10, str(second))
            twice = process_file(str(second), skip, rows, points)
            np.testing.assert_allclose(twice, array * 10, atol=1e-10, rtol=1e-10)
            self.assertEqual(int(np.count_nonzero(twice < 0)), 400)

    def test_whitespace_D_exponents_and_subsequent_augmentation(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "CHGCAR"
            source.write_text(
                header((2, 2, 2))
                + " 0.10D+00\t-0.20D+00   0.30E+00\n"
                + "\n"
                + " -0.40E+00  0.00E+00\n"
                + " 0.60E+00   -0.70E+00  0.80E+00\n"
                + "\naugmentation occupancies 1 2 3\n",
                encoding="utf-8")
            _, _, points, _, _, _, skip, rows = read_file_info(str(source))
            actual = process_file(str(source), skip, rows, points)
            np.testing.assert_allclose(actual,
                [0.1, -0.2, 0.3, -0.4, 0.0, 0.6, -0.7, 0.8])

    def test_truncated_grid_fails_closed(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "CHGCAR"
            source.write_text(header((2, 2, 2)) + "0.1 -0.2 0.3\n")
            with self.assertRaisesRegex(ValueError, "Truncated"):
                process_file(str(source), 10, 2, 8)

    def test_nonnumeric_and_nonfinite_samples_fail_closed(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "CHGCAR"
            for item, message in (("bad", "Invalid"), ("nan", "Non-finite"),
                                  ("inf", "Non-finite")):
                source.write_text(header((1, 1, 1)) + item + "\n")
                with self.subTest(item=item):
                    with self.assertRaisesRegex(ValueError, message):
                        process_file(str(source), 10, 1, 1)


if __name__ == "__main__":
    unittest.main()
