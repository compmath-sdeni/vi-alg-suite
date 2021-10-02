import numpy as np


class ConvexSetConstraintsException(Exception):
    """Exception raised for errors in convex set constraints classes.

        Attributes:
            method -- method in which the error occurred
            message -- explanation of the error
        """

    def __init__(self, method, message):
        self.method = method
        self.message = message


class ConvexSetConstraints:
    def __init__(self):
        self.zero_delta = 1e-12

    def isIn(self, x: np.ndarray) -> bool:
        pass

    def getSomeInteriorPoint(self) -> np.ndarray:
        pass

    def project(self, x: np.ndarray) -> np.ndarray:
        pass

    def getDistance(self, x: np.array) -> float:
        return np.linalg.norm(x - self.project(x))

    def getDim(self):
        raise ConvexSetConstraintsException('getDim', 'Not implemented!')

    def toString(self):
        return "ConvexSet"

    def __str__(self):
        return self.toString()
