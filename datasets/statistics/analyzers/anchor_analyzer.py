"""
===========================================================
Anchor Analyzer

Even though YOLOv11 is anchor-free,
this module analyzes object sizes.

Useful for

• Image size recommendation
• Multi-scale recommendation
• Dataset difficulty
• Detector comparison
===========================================================
"""

from statistics import median

from datasets.statistics.analyzers.base_analyzer import BaseAnalyzer


class AnchorAnalyzer(BaseAnalyzer):

    SMALL = 0.01
    LARGE = 0.10

    ###########################################################

    def analyze(self, dataset, statistics):

        self.log("Analyzing object size distribution...")

        widths = []
        heights = []
        areas = []
        ratios = []

        anchor = statistics.anchors

        #######################################################

        for split in dataset.splits:

            labels = dataset.get_labels(split)

            for label in labels:

                with open(label) as f:

                    for line in f:

                        line = line.strip()

                        if not line:

                            continue

                        values = line.split()

                        if len(values) < 5:

                            continue

                        _, _, _, w, h = map(float, values[:5])

                        widths.append(w)

                        heights.append(h)

                        areas.append(w * h)

                        ratios.append(w / h if h else 0)

        #######################################################

        if not widths:

            return

        anchor.total_boxes = len(widths)

        anchor.average_width = sum(widths) / len(widths)

        anchor.average_height = sum(heights) / len(heights)

        anchor.average_area = sum(areas) / len(areas)

        anchor.average_aspect_ratio = (
            sum(ratios) / len(ratios)
        )

        anchor.median_width = median(widths)

        anchor.median_height = median(heights)

        anchor.median_area = median(areas)

        #######################################################
        # Object Size Categories
        #######################################################

        for area in areas:

            if area < self.SMALL:

                anchor.small_objects += 1

            elif area < self.LARGE:

                anchor.medium_objects += 1

            else:

                anchor.large_objects += 1

        #######################################################
        # Percentiles
        #######################################################

        widths = sorted(widths)
        heights = sorted(heights)
        areas = sorted(areas)

        def percentile(values, p):

            index = int(
                p / 100 * (len(values) - 1)
            )

            return values[index]

        for p in [5, 25, 50, 75, 95]:

            anchor.width_percentiles[str(p)] = percentile(
                widths,
                p,
            )

            anchor.height_percentiles[str(p)] = percentile(
                heights,
                p,
            )

            anchor.area_percentiles[str(p)] = percentile(
                areas,
                p,
            )

        #######################################################
        # Recommendations
        #######################################################

        if anchor.small_objects > anchor.total_boxes * 0.7:

            anchor.recommendations.append(
                "Increase image size to 960 or 1280."
            )

            anchor.recommendations.append(
                "Enable multi-scale training."
            )

        if anchor.average_aspect_ratio > 2:

            anchor.recommendations.append(
                "Objects are elongated."
            )

        self.log("Anchor analysis complete.")