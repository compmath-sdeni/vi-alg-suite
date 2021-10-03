import numpy as np
from constraints.convex_set_constraint import ConvexSetConstraints, ConvexSetConstraintsException
from methods.projections.simplex_projection_prom import euclidean_proj_l1ball


class L1Ball(ConvexSetConstraints):
    def __init__(self, n: int, b: float = 1.0):
        super().__init__()
        self.n: int = n
        self.b: float = b

    def isIn(self, x: np.ndarray) -> bool:
        return np.sum(np.fabs(x)) <= self.b

    def getSomeInteriorPoint(self) -> np.ndarray:
        res: np.ndarray = np.random.rand(self.n)*self.b
        it = 0
        while not self.isIn(res) and it < self.max_projection_iters:
            max_idx = res.argmax()
            res[max_idx] *= 0.5
            it += 1

        if self.isIn(res):
            return res
        else:
            raise ConvexSetConstraintsException("L1 ball getSomeInteriorPoint", "Iterations limit reached!")

    def getDim(self):
        return self.n

    def project(self, x: np.ndarray) -> np.ndarray:
        if self.isIn(x):
            return x
        else:
            # using prom algo
            return euclidean_proj_l1ball(x, s=self.b)

            # using self-coded algho
            # res: np.ndarray = x.copy()
            #
            # if not self.isIn(res):
            #     neg: np.ndarray = res < 0
            #     res[neg] *= -1
            #     SimplexProj.doInplace(res, self.b)
            #     res[neg] *= -1
            #
            # return res

    def toString(self):
        return "{0}d L1 ball R={1}".format(self.n, self.b)
