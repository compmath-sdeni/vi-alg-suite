import numpy as np


class ConvexSetConstraints:
    def isIn(self, x: np.ndarray) -> bool:
        pass

    def getSomeInteriorPoint(self) -> np.ndarray:
        pass

    def project(self, x: np.ndarray) -> np.ndarray:
        pass
