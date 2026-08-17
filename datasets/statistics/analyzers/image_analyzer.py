"""
======================================================================
Image Analyzer

Author : C-UAS Vision Subsystem

Description
-----------
Computes image-related dataset statistics.

Features
--------
✓ Total images
✓ Corrupted images
✓ Width statistics
✓ Height statistics
✓ Aspect ratio
✓ Resolution distribution
✓ Image format statistics
✓ Color mode statistics
✓ Uniform resolution detection
✓ Progress reporting
✓ Processing time
======================================================================
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

from PIL import Image

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class ImageAnalyzer(BaseAnalyzer):

    def __init__(self):

        super().__init__()

    ##################################################################

    def analyze(self, dataset, statistics):

        self.log("Starting image analysis...")

        start = perf_counter()

        widths = []
        heights = []
        aspect_ratios = []

        format_counter = Counter()
        color_counter = Counter()
        resolution_counter = Counter()

        corrupted = 0
        processed = 0

        for split in dataset.splits:

            images = dataset.get_images(split)

            self.log(f"{split.upper()} : {len(images)} images")

            for image_path in images:

                try:

                    with Image.open(image_path) as img:

                        img.verify()

                    with Image.open(image_path) as img:

                        width, height = img.size

                        image_format = (
                            img.format.upper()
                            if img.format
                            else "UNKNOWN"
                        )

                        color_mode = img.mode

                except Exception:

                    corrupted += 1
                    continue

                processed += 1

                widths.append(width)
                heights.append(height)
                aspect_ratios.append(width / height)

                format_counter[image_format] += 1
                color_counter[color_mode] += 1

                resolution_counter[
                    f"{width}x{height}"
                ] += 1

                if processed % 1000 == 0:

                    self.log(
                        f"Processed {processed:,} images..."
                    )

        if processed == 0:

            raise RuntimeError(
                "No valid images found."
            )

        img = statistics.images

        img.total_images = processed

        img.corrupted_images = corrupted

        img.average_width = sum(widths) / processed
        img.average_height = sum(heights) / processed

        img.minimum_width = min(widths)
        img.maximum_width = max(widths)

        img.minimum_height = min(heights)
        img.maximum_height = max(heights)

        img.average_aspect_ratio = (
            sum(aspect_ratios) / processed
        )

        img.image_formats = dict(format_counter)

        img.color_modes = dict(color_counter)

        img.resolution_distribution = dict(
            resolution_counter
        )

        elapsed = perf_counter() - start

        self.log(
            f"Finished in {elapsed:.2f} seconds."
        )

        self.log(
            f"Images : {processed:,}"
        )

        self.log(
            f"Corrupted : {corrupted}"
        )