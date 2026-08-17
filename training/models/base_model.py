"""
===========================================================
Base Detection Model
===========================================================
"""

from abc import ABC
from abc import abstractmethod

from training.config import TrainingConfig


class BaseModel(ABC):

    """
    Abstract interface implemented by every detector.
    """

    def __init__(self, config: TrainingConfig):

        self.config = config

    ##############################################################

    @abstractmethod
    def load(self):

        """Load weights"""

        raise NotImplementedError

    ##############################################################

    @abstractmethod
    def summary(self):

        """Print model information"""

        raise NotImplementedError
