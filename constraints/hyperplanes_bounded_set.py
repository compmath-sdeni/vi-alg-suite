import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints, ConvexSetConstraintsException


class HyperplanesBoundedSet(ConvexSetConstraints):
    # e.g. [([1,1],2),([-1,-1],-1)] stands for x+y<=2 and -x + -y <=-1 <=> x+y>=1

    def __init__(self, *, spaceDim: int, constraints: list, eps: float = 1e-10, maxProjectIters: int = 1000):
        """
        :param spaceDim: working space dimension (n for Hn)
        :param constraints:
            list of hyperplanes as tuples of c and b, for <c,x> <= b notation.
            # e.g. [([1, 1], 2), ([-1, -1], -1)] stands for x+y<=2 and -x-y<=-1 <=> x+y>=1
        :param eps: constraint check allowed error
        """

        self.spaceDim = spaceDim
        self.constraints = [[np.array(c), b] for c, b in constraints]
        self.eps = eps
        self.maxProjectIters = maxProjectIters

        for c, b in self.constraints:
            if len(c) != self.spaceDim:
                raise Exception(
                    "Hyperrectangle - bad hyperplane normal vector dimensions: {0} (need {1})".format(len(c),
                                                                                                      self.spaceDim))

    def _isConstraintSatisfied(self, x: np.array, constraintIndex: int):
        return np.dot(self.constraints[constraintIndex][0], x) <= self.constraints[constraintIndex][1]

    def _getFailedHyperplane(self, x: np.ndarray):
        return next(filter(lambda constr: np.dot(constr[0], x) - self.eps >= constr[1], self.constraints),
                    False)

    def isIn(self, x: np.ndarray) -> bool:
        failed_constr = self._getFailedHyperplane(x)
        return failed_constr == False

    def getSomeInteriorPoint(self) -> np.array:
        res = self.project(np.ones(self.spaceDim))

        # TODO: try to shift res...
        return res

    def project(self, x: np.array) -> np.array:
        # project to Hn = (c,x)<=b
        # Px = x + (b - <c,x>)*c/<c,c>

        failed_hyperplane = self._getFailedHyperplane(x)
        if not failed_hyperplane:
            return x

        t = x.copy()
        k = 1
        n = len(self.constraints)

        while k < self.maxProjectIters:
            # print("t: {0}; k:{1}; p: {2}; failed: {3}".format(t, k, p, failedConstr))
            # pp = np.copy(p);
            # for i in range(n):
            #     constr = self.constraints[i]
            #     p[i] = t/lam + pp[i] - ((constr[1] - np.dot(constr[0], (t+lam*pp[i]))) * constr[0]) / (np.dot(constr[0], constr[0]) * lam)
            #
            # t = x - np.sum(p, axis=0)

            print("t: {0}; k:{1}; failed: {2}".format(t, k, failed_hyperplane))
            for i in range(n):
                constr = self.constraints[i]
                if np.dot(constr[0], x) - self.eps >= constr[1]:
                    t = t + ((constr[1] - np.dot(constr[0], t)) * constr[0]) / np.dot(constr[0], constr[0])
                    # print("Projected to {0}; t = {1}".format(constr[0], t))

            t = (1.0 / (k ** 2)) * x + (1.0 - 1.0 / (k ** 2)) * t

            # t = t + ((failedConstr[1] - np.dot(failedConstr[0], t)) * failedConstr[0]) / np.dot(
            #     failedConstr[0], failedConstr[0])

            failed_hyperplane = self._getFailedHyperplane(x)
            if not failed_hyperplane:
                break

            k += 1

        # print('Projected in {0} iters. Res: {1}'.format(k, t))

        if self.isIn(t):
            return t
        else:
            raise ConvexSetConstraintsException('HyperplanesBoundedSet project',
                                                'Maximum number of iterations reached!')

    def constraintToString(self, constraint):
        c, b = constraint
        eps = 0.00000000000001
        return "".join([
            (
                (('-' if xi < 0 else ' ') if c[0] >= 0 else ('-' if xi > 0 else ' ')) if i == 0
                else ((' - ' if xi < 0 else (' + ' if xi > 0 else ' ')) if c[0] >= 0 else (
                    ' - ' if xi > 0 else (' + ' if xi < 0 else ' ')))
            )
            + ((str(abs(xi)) if (abs(xi) > eps and abs(xi - 1) > eps and abs(xi + 1) > eps) else ''))
            + (('x' + str(i)) if abs(xi) > eps else ' ')
            for i, xi in enumerate(c)]) + (' <= ' if c[0] >= 0 else ' >= ') + str(b * (-1 if c[0] < 0 else 1))

    def toString(self):
        # boundsStr = "\n".join(["".join([((('+' if i>0 else ' ') if b>0 else '-') if i>0 and xi>0 else ('-' if b>0 else ('+' if i>0 else ' ')) if xi<0 else ' ') + (str(abs(xi)) if abs(xi) != 1 and xi!=0 else '') + 'x' + str(i) for i, xi in enumerate(c)]) + ('<=' if b>0 else '>=') + str(abs(b)) for c, b in [bound for bound in self.constraints]])

        boundsStr = "\n".join([self.constraintToString(constr) for constr in self.constraints])

        # boundsStr = "\n".join(["".join([
        #                                     (
        #                                         (('-' if xi < 0 else ' ') if b >= 0 else ('-' if xi > 0 else ' ')) if i == 0
        #                                         else ((' - ' if xi < 0 else ' + ') if b >= 0 else (' - ' if xi > 0 else ' + '))
        #                                     )
        #                                     + ((str(abs(xi)) if (abs(xi) > 0.0000001 and abs(xi - 1) > 0.00000001 and abs(xi + 1) > 0.00000001) else ' '))
        #                                    + 'x' + str(i) for i, xi in
        #                                 enumerate(c)]) + ('<=' if b > 0 else '>=') + str(abs(b)) for c, b in
        #                        [bound for bound in self.constraints]])
        return "Space dim: {0}; bounds:\n{1}".format(self.spaceDim, boundsStr)
