"""
======================================================================
Base Analyzer

Author : C-UAS Vision Subsystem

Description
-----------
Abstract base class for all dataset statistics analyzers.

Every analyzer must inherit from BaseAnalyzer and implement:

    analyze(dataset_manager, statistics)

This guarantees a consistent interface throughout the statistics
framework.
======================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from datasets.dataset_manager import DatasetManager
from datasets.statistics.models import DatasetStatisticsModel


class BaseAnalyzer(ABC):
    """
    Base class for all statistics analyzers.
    """

    def __init__(self):

        self.name = self.__class__.__name__

    ##################################################################

    @abstractmethod
    def analyze(
        self,
        dataset: DatasetManager,
        statistics: DatasetStatisticsModel,
    ) -> None:
        """
        Analyze the dataset and update the statistics model.

        Parameters
        ----------
        dataset : DatasetManager
            Dataset manager instance.

        statistics : DatasetStatisticsModel
            Statistics model to populate.
        """
        raise NotImplementedError

    ##################################################################

    def log(self, message: str):

        print(f"[{self.name}] {message}")