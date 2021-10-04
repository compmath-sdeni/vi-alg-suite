import math
import os
from typing import List

import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints, ConvexSetConstraintsException


class ConvexSetsIntersection(ConvexSetConstraints):
    def __init__(self, sets: List[ConvexSetConstraints], *, max_projection_iters: int = None,
                 projection_eps: float = None):

        super().__init__(
            max_projection_iters=max_projection_iters if max_projection_iters is not None else len(sets) * 1000
        )

        self.sets = sets
        self.projection_eps = projection_eps if projection_eps is not None else self.zero_delta

    def isIn(self, x: np.array) -> bool:
        for s in self.sets:
            if not s.isIn(x):
                return False

        return True

    def getDim(self):
        return self.sets[0].getDim()

    def getSomeInteriorPoint(self) -> np.array:
        x = self.sets[0].getSomeInteriorPoint()
        i = 0

        is_in = False
        while i < self.max_projection_iters and not is_in:
            is_in = True
            i += 1
            for s in self.sets:
                if not s.isIn(x):
                    is_in = False
                    x = s.project(x)

        if is_in:
            return x
        else:
            raise ConvexSetConstraintsException('getSomeInteriorPoint', 'Maximum number of iterations reached!')

    def project(self, x: np.array) -> np.array:
        i = 0
        is_in = False
        xn = x.copy()
        pn = np.zeros_like(x)
        qn = np.zeros_like(x)

        while i < self.max_projection_iters and not is_in:
            is_in = True
            i += 1

            # print(f"{i}; {xn}; Dist: {np.linalg.norm(x - xn)}")

            yn = self.sets[0].project(xn + pn)
            pn = xn + pn - yn
            xn = self.sets[1].project(yn + qn)
            qn = yn + qn - xn

            for s in self.sets:
                if not s.isIn(xn):
                    if s.getDistance(xn) >= self.projection_eps:
                        # print(f"{xn} not in {s}. Distance: {s.getDistance(xn)}")
                        is_in = False
                # else:
                #     print(f"In {s}")

        if is_in:
            return xn
        else:
            raise ConvexSetConstraintsException('ConvexSetsIntersection project',
                                                'Maximum number of iterations reached!')

    def project2(self, x: np.array) -> np.array:
        i = 0
        is_in = False
        xn = x.copy()

        while i < self.max_projection_iters and not is_in:
            alp = 1./math.pow(i+1, 0.5)

            is_in = True
            i += 1

            print(f"{i}; {alp}; {xn}")

            for s in self.sets:
                if not s.isIn(xn):
                    print(f"Not in {s}. Distance: {s.getDistance(xn)}")
                    is_in = False
                    xn = xn*alp + (1-alp)*s.project(xn)
                else:
                    print(f"In {s}")

        if is_in:
            return xn
        else:
            raise ConvexSetConstraintsException('ConvexSetsIntersection project',
                                                'Maximum number of iterations reached!')

    def saveToDir(self, path:str):
        for i, constr in enumerate(self.sets):
            ind_path = os.path.join(path, f"c{i}")
            os.makedirs(ind_path, exist_ok=True)
            constr.saveToDir(ind_path)

    def toString(self):
        return f"ConvexSetIntersection-{len(self.sets)}-{self.getDim()}\n{[str(s) for s in self.sets]}"