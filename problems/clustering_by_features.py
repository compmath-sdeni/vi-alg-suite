import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import scale
from problems.problem import Problem
from tools.print_utils import vectorToString
import os


class ClusteringByFeaturesProblem(Problem):
    def __init__(self, *, K: int, Y: np.ndarray, hard: bool = False, m: int = 0, uTest: np.ndarray = None, vTest: np.ndarray = None,
                 u0: np.ndarray = None, v0: np.ndarray = None):
        """
        Determine K clusters by p features of n entities, passed as Y matrix.
        Model order is m.
        """

        super().__init__(xtest=None, x0=u0)

        self.m = m if m is not None else 0
        self.K = K
        self.Y = Y
        self.hard = hard
        self.n = Y.shape[0]
        self.p = Y.shape[1]
        self.uTest = uTest
        self.vTest = vTest
        self.u0 = u0 if u0 is not None else np.full((self.K, self.n), 0.5, float)
        self.v0 = v0 if v0 is not None else np.full((self.K, self.p), 0.5, float)

        if self.m == 0:
            def ei(y: float, v: float, u: float) -> float:
                return (y - v * u) ** 2
        else:
            def ei(y: float, v: float, u: float) -> float:
                return (u ** m) * ((y - v * u) ** 2)

        self.ei = ei

    def randomizeInitialValues(self, K: int = None):
        if K is not None:
            self.K = K

        self.v0 = np.random.ranf((self.K, self.p))
        self.u0 = np.random.ranf((self.K, self.n))

    @classmethod
    def createRandom(cls, *, K: int = 2, n: int = 3, p: int = 2, m: int = 1, div: float = 0.5):

        v = np.random.ranf((K, p))
        shift = [s for s in zip(np.random.randint(0, K, n), np.random.ranf((n, p)) * div - div * 0.5)]
        Y = np.array([v[s[0]] + s[1] for s in shift])

        max = np.max(Y, 0)
        min = np.min(Y, 0)

        Y = (Y - min) / (max - min)
        v = (v - min) / (max - min)

        u = np.zeros((K, n), float)
        for i, j in [(s[1][0], s[0]) for s in enumerate(shift)]:
            u[i, j] = 1

        vshift = np.random.ranf((K, p))-0.5
        ushift = np.random.ranf((K, n))-0.5
        v0 = v + (vshift * (div*0.5))
        max = np.max(v0, 0)
        min = np.min(v0, 0)
        v0 = (v0 - min) / (max - min)

        u0 = u + (ushift * (div*0.5))
        max = np.max(u0, 0)
        min = np.min(u0, 0)
        u0 = (u0 - min) / (max - min)

        return ClusteringByFeaturesProblem(K=K, Y=Y, vTest=v, uTest=u, m=m, u0=u0, v0=v0)

    @classmethod
    def createRandomFuzzy(cls, *, K: int = 2, p: int = 2, d: float = 0.1, densePercent: float = 0.75, nMin: int = 3, nMax: int = 6, m: int = 2, doScale: bool = True):
        nk = np.random.randint(nMin, nMax, K)  # number of entities around each cluster
        n = nk.sum()

        print("createRandomFuzzy:\nnk={0}; n={1}".format(nk, n))

        v = np.random.uniform(-100, 100, (K, p))  # cluster centers
        o = np.average(v, 0)  # origin

        print("v:\n{0}\no: {1}".format(vectorToString(v), vectorToString(o)))

        a = np.array([v * (1.0 - d), v * (1.0 + d)])  # bounds of 'tight bound points area'
        b = np.array([np.broadcast_to(o, (K, p)), v])  # bounds of 'loose bound points area'

        print("a:\n{0}\nb: {1}".format(a, b))

        Y = np.empty((0, p))
        u = np.zeros((K, n), float)

        for i in range(K):
            y1 = np.array([np.random.uniform(a[0, i], a[1, i]) for j in range(int(nk[i] * densePercent))])
            y2 = np.array([np.random.uniform(b[0, i], b[1, i]) for j in range(nk[i] - int(nk[i] * densePercent))])
            u[i, len(Y):len(Y)+nk[i]] = 1

            Y = np.append(Y, y1, 0)
            Y = np.append(Y, y2, 0)

        print("Y:\n{0}".format(vectorToString(Y)))

        v -= o
        Y -= o

        print("Shifted Y:\n{0}\nShifted v:\n{1}".format(vectorToString(Y), vectorToString(v)))

        if doScale:
            # noinspection PyShadowingBuiltins
            max = np.max(Y, 0)
            # noinspection PyShadowingBuiltins
            min = np.min(Y, 0)

            Y = (Y - min) / (max - min)
            v = (v - min) / (max - min)

        return ClusteringByFeaturesProblem(K=K, Y=Y, vTest=v.copy(), uTest=u, m=m, u0=u, v0=v.copy())

    def ECM(self, v: np.ndarray, u: np.ndarray) -> float:
        D = distance.cdist(v, self.Y)
        s = 0.0

        for k in np.arange(self.K):
            for i in np.arange(self.n):
                s += u[k, i] * D[k, i]
        return s

    def ECPM(self, v: np.ndarray, u: np.ndarray) -> float:
        s = 0.0
        for k in np.arange(self.K):
            for i in np.arange(self.n):
                for h in np.arange(self.p):
                    s += self.ei(self.Y[i, h], v[k, h], u[k, i])
        return s

    def F(self, x):
        # noinspection PyShadowingNames
        def v(x: np.ndarray, k: int, h: int) -> float:
            return x[k * self.p + h]

        # noinspection PyShadowingNames
        def u(x: np.ndarray, k: int, i: int) -> float:
            return x[k * self.n + i]

        s = 0
        k = 0
        while k < self.K:
            i = 0
            while i < self.n:
                h = 0
                while h < self.p:
                    s += self.ei(self.Y[i, h], v(x, k, h), u(x, k, i))
                    h += 1
                i += 1
            k += 1

    def saveToFile(self, pathPrefix: str = None) -> str:
        basePath = super().getSavePath(pathPrefix)

        np.savetxt("{0}/{1}".format(basePath, 'Y.txt'), self.Y)
        np.savetxt("{0}/{1}".format(basePath, 'k-m.txt'), np.array([self.K, self.m]))
        np.savetxt("{0}/{1}".format(basePath, 'u0.txt'), self.u0)
        np.savetxt("{0}/{1}".format(basePath, 'v0.txt'), self.v0)
        np.savetxt("{0}/{1}".format(basePath, 'uTest.txt'), self.uTest)
        np.savetxt("{0}/{1}".format(basePath, 'vTest.txt'), self.vTest)
        return basePath

    @classmethod
    def loadFromFile(self, path: str):

        Y = np.loadtxt("{0}/{1}".format(path, 'Y.txt'))
        tmp = np.loadtxt("{0}/{1}".format(path, 'k-m.txt'), dtype=int)
        K, m = tmp
        u0 = np.loadtxt("{0}/{1}".format(path, 'u0.txt'))
        v0 = np.loadtxt("{0}/{1}".format(path, 'v0.txt'))
        uTest = np.loadtxt("{0}/{1}".format(path, 'uTest.txt'))
        vTest = np.loadtxt("{0}/{1}".format(path, 'vTest.txt'))

        return ClusteringByFeaturesProblem(K=K, Y=Y, m=m, u0=u0, v0=v0, uTest=uTest, vTest=vTest)
