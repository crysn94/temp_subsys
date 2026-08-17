"""
===========================================================
Bounding Box Analyzer
===========================================================
"""

from collections import Counter

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class BoundingBoxAnalyzer(BaseAnalyzer):

    SMALL_AREA = 0.02
    LARGE_AREA = 0.15

    EDGE_MARGIN = 0.05

    ########################################################

    def analyze(self, dataset, statistics):

        self.log("Starting bounding box analysis...")

        bbox = statistics.bounding_boxes

        widths = []
        heights = []
        areas = []
        ratios = []

        occupancy = []

        center_x_counter = Counter()
        center_y_counter = Counter()

        small = 0
        medium = 0
        large = 0

        edge = 0

        total_boxes = 0

        ####################################################

        for split in dataset.splits:

            labels = dataset.get_labels(split)

            for label_file in labels:

                with open(label_file) as f:

                    for line in f:

                        line = line.strip()

                        if not line:
                            continue

                        values = line.split()

                        if len(values) < 5:
                            continue

                        _, x, y, w, h = map(float, values[:5])

                        area = w * h

                        ratio = w / h if h > 0 else 0

                        widths.append(w)
                        heights.append(h)

                        areas.append(area)

                        ratios.append(ratio)

                        occupancy.append(area)

                        total_boxes += 1

                        ####################################

                        if area < self.SMALL_AREA:

                            small += 1

                        elif area < self.LARGE_AREA:

                            medium += 1

                        else:

                            large += 1

                        ####################################

                        if (

                            x < self.EDGE_MARGIN
                            or y < self.EDGE_MARGIN
                            or x > 1 - self.EDGE_MARGIN
                            or y > 1 - self.EDGE_MARGIN

                        ):

                            edge += 1

                        ####################################

                        center_x_counter[
                            f"{round(x,1):.1f}"
                        ] += 1

                        center_y_counter[
                            f"{round(y,1):.1f}"
                        ] += 1

        ####################################################

        bbox.total_boxes = total_boxes

        if total_boxes == 0:

            return

        bbox.average_width = sum(widths) / total_boxes

        bbox.average_height = sum(heights) / total_boxes

        bbox.average_area = sum(areas) / total_boxes

        bbox.average_aspect_ratio = (
            sum(ratios) / total_boxes
        )

        bbox.minimum_width = min(widths)

        bbox.maximum_width = max(widths)

        bbox.minimum_height = min(heights)

        bbox.maximum_height = max(heights)

        bbox.smallest_area = min(areas)

        bbox.largest_area = max(areas)

        bbox.average_occupancy = (
            sum(occupancy) / total_boxes
        )

        bbox.small_objects = small

        bbox.medium_objects = medium

        bbox.large_objects = large

        bbox.edge_objects = edge

        bbox.center_x_distribution = dict(center_x_counter)

        bbox.center_y_distribution = dict(center_y_counter)

        self.log("Bounding box analysis complete.")