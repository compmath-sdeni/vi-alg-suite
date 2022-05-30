import os
from datetime import datetime
from typing import Union, Dict, Optional

import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints
from utils.print_utils import *


class Problem:
    defDataDir = 'data'

    def __init__(self, *,
                 xtest: Union[np.ndarray, float] = None,
                 x0: Union[np.ndarray, float] = None,
                 C: ConvexSetConstraints = None,
                 hr_name: str = None,
                 x_dim: int = None,
                 lam_override: float = None,
                 lam_override_by_method: dict = None,
                 zero_cutoff: float = None):

        self._x0: Union[np.ndarray, float] = x0
        self.C = C
        self.xtest: Union[np.ndarray, float] = xtest
        self.hr_name: str = hr_name
        self.lam_override = lam_override
        self.lam_override_by_method = lam_override_by_method
        self.x_dim = x_dim if x_dim is not None else x0.shape[0]
        self.zero_cutoff = zero_cutoff

    @property
    def x0(self) -> Union[np.ndarray, float]:
        return self._x0

    def F(self, x: Union[np.ndarray, float]):
        pass

    def GradF(self, x: Union[np.ndarray, float]):
        pass

    def A(self, x: Union[np.ndarray, float]):
        pass

    def updateStructure(self, x: Union[np.ndarray, float]):
        pass

    def getSavePath(self, path_prefix: str = None) -> str:
        """

        :type path_prefix: str # path prefix without trailing slash. If None, default data dir (e.g. 'data') is used
        :rtype: str
        :return: full path to dir with data
        """

        if path_prefix is None:
            path_prefix = self.defDataDir
        base_path = "{0}/{1}/{2:%y_%m_%d_%H_%M_%S}".format(path_prefix, self.__class__.__name__, datetime.now())
        os.makedirs(base_path, exist_ok=True)

        return base_path

    def saveToDir(self, *, path_to_save: str = None):
        if path_to_save is None:
            path_to_save = self.getSavePath()

        if self.C is not None:
            constr_path = os.path.join(path_to_save, 'constraints')
            os.makedirs(constr_path, exist_ok=True)
            self.C.saveToDir(constr_path)

        return path_to_save

    def XToString(self, x: np.array):
        return vectorToString(x)

    def FValToString(self, v: float):
        return scalarToString(v)

    def GetErrorByTestX(self, xstar: Union[np.array, np.ndarray, float]) -> float:
        return np.dot((self.xtest - xstar), (self.xtest - xstar)) if self.xtest is not None else xstar

    def GetHRName(self):
        return self.hr_name if self.hr_name is not None else self.__class__.__name__

    def GetFullDesc(self):
        res = self.GetHRName()
        return res

    def GetLambdaOverride(self):
        return self.lam_override

    def GetExtraIndicators(self, x: Union[np.ndarray, float], *, averaged_x: np.ndarray = None, final: bool = False) -> Optional[Dict]:
        pass
