from abc import abstractmethod
from typing import Dict
import numpy as np


class BaseController:

    @abstractmethod
    def step(self, obs:np.ndarray, info:Dict) -> int | float | np.ndarray:
        """
        Compute the action to be taken by the controller based on the current observation and additional information.

        Args:
            obs (np.ndarray): The current observation from the environment.
            info (Dict): Additional information provided by the environment.

        Returns:
            int | float | np.ndarray: The action to be taken by the controller.
        """
        return 0.0