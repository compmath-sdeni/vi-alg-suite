import numpy as np

from constraints.convex_set_constraint import ConvexSetConstraints


class HyperplanesBoundedSet(ConvexSetConstraints):
    # e.g. [([1,1],2),([-1,-1],-1)] stands for x+y<=2 and -x + -y <=-1 <=> x+y>=1

    def __init__(self, *, spaceDim: int, constraints: list, eps:float = 1e-10, maxProjectIters:int = 1000):
        """
        :param spaceDim: working space dimention (n for Hn)
        :param constraints:
            list of hyperplanes as tuples of c and b, for <c,x> <= b notation.
            # e.g. [([1, 1], 2), ([-1, -1], -1)] stands for x+y<=2 and -x-y<=-1 <=> x+y>=1
        :param eps: constraint check allowed error
        """

        self.spaceDim=spaceDim
        self.constraints = [[np.array(c), b] for c, b in constraints]
        self.eps = eps
        self.maxProjectIters = maxProjectIters

        for c, b in self.constraints:
            if len(c) != self.spaceDim:
                raise Exception("Hyperrectangle - bad hyperplane normal vector dimentions: {0} (need {1})".format(len(c), self.spaceDim))

        print("HyperplanesBoundedSet created. {0}".format(self.toString()))

    def _isConstraintSatisfied(self, x:np.array, constraintIndex:int):
        return (np.dot(self.constraints[constraintIndex][0], x) <= self.constraints[constraintIndex][1])

    def isIn(self, x:np.array) -> bool:
        failedConstr = next(filter(lambda constr: np.dot(constr[0], x) - self.eps >= constr[1], self.constraints), False)
        return failedConstr == False, failedConstr

    def getSomeInteriorPoint(self) -> np.array:
        #TODO: how to get some interior point?
        return None

    def project(self,x:np.array) -> np.array:
        # project to Hn = (c,x)<=b
        # Px = x + (b - <c,x>)*c/<c,c>

        isIn, failedConstr = self.isIn(x)
        if isIn:
            return x

        t = x.copy()
        k = 1
        n = len(self.constraints)

        # lam =  2
        # I = np.ones(self.spaceDim)
        # p = np.array([np.random.rand(self.spaceDim) for i in range(n)])
        # print('p: {0}; I: {1}; lam:{2}'.format(p, I, lam))
        # t = x - np.sum(p, axis=0)

        isIn, failedConstr = self.isIn(t)

        while k < self.maxProjectIters:
            # print("t: {0}; k:{1}; p: {2}; failed: {3}".format(t, k, p, failedConstr))
            # pp = np.copy(p);
            # for i in range(n):
            #     constr = self.constraints[i]
            #     p[i] = t/lam + pp[i] - ((constr[1] - np.dot(constr[0], (t+lam*pp[i]))) * constr[0]) / (np.dot(constr[0], constr[0]) * lam)
            #
            # t = x - np.sum(p, axis=0)

            print("t: {0}; k:{1}; failed: {2}".format(t, k, failedConstr))
            for i in range(n):
                constr = self.constraints[i]
                if np.dot(constr[0], x) - self.eps >= constr[1]:
                    t = t + ((constr[1] - np.dot(constr[0], t)) * constr[0]) / np.dot(constr[0], constr[0])
                    #print("Projected to {0}; t = {1}".format(constr[0], t))

            t = (1.0/(k**2))*x + (1.0-1.0/(k**2))*t

            # t = t + ((failedConstr[1] - np.dot(failedConstr[0], t)) * failedConstr[0]) / np.dot(
            #     failedConstr[0], failedConstr[0])

            isIn, failedConstr = self.isIn(t)
            k+=1

        #print('Projected in {0} iters. Res: {1}'.format(k, t))

        if k<self.maxProjectIters:
            return t
        else:
            return t
            #raise Exception("HyperplanesBoundedSet projection failed! Iters: {0}; Last result: {1}".format(k, vectortostring(t)))

    def constraintToString(self, constraint):
        c, b = constraint
        eps = 0.00000000000001
        return "".join([
                    (
                        (('-' if xi < 0 else ' ') if c[0] >= 0 else ('-' if xi > 0 else ' ')) if i == 0
                        else ((' - ' if xi < 0 else (' + ' if xi > 0 else ' ')) if c[0] >= 0 else (' - ' if xi > 0 else (' + ' if xi < 0 else ' ')))
                    )
                    + ((str(abs(xi)) if (abs(xi) > eps and abs(xi - 1) > eps and abs(xi + 1) > eps) else ''))
                    + (('x' + str(i)) if abs(xi) > eps else ' ')
                    for i, xi in enumerate(c)]) + (' <= ' if c[0] >= 0 else ' >= ') + str(b * (-1 if c[0]<0 else 1))
    def toString(self):
        #boundsStr = "\n".join(["".join([((('+' if i>0 else ' ') if b>0 else '-') if i>0 and xi>0 else ('-' if b>0 else ('+' if i>0 else ' ')) if xi<0 else ' ') + (str(abs(xi)) if abs(xi) != 1 and xi!=0 else '') + 'x' + str(i) for i, xi in enumerate(c)]) + ('<=' if b>0 else '>=') + str(abs(b)) for c, b in [bound for bound in self.constraints]])

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
