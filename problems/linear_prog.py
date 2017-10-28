import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from problems.viproblem import VIProblem
from problems.visual_params import VisualParams
from constraints.convex_set_constraint import ConvexSetConstraints


class LinearProgProblem(VIProblem):

    def __init__(self, *, A:np.array = np.array([[1,0], [0,1]]), b:np.array = np.array([1,1]), c = np.array([1,1]), x0 = np.array([1,1,1,1]), vis:VisualParams = VisualParams(), defaultProjection: np.array = None, xtest: np.array = None):
        self.A = A
        self.b = b
        self.c = c
        self.x0 = x0

        self.n = c.shape[0]
        self.m = b.shape[0]

        self.vis = vis
        self.defaultProjection = defaultProjection if defaultProjection is not None else np.zeros(self.n)
        self.xtest = xtest if xtest is not None else np.zeros(self.n+self.m)

    def y(self, u:np.array) -> np.array:
        return u[self.n:]

    def x(self, u:np.array) -> np.array:
        return u[:self.n]

    def F(self, u:np.array) -> float:
        return np.dot(self.c, self.x(u))
        #return np.dot(self.c, self.x(u) + np.dot(self.y(u), np.dot(self.A, self.x(u))))

    def GradF(self, u:np.array) -> float:
        return np.array([*(np.dot(np.transpose(self.A), self.y(u))+self.c),*(self.b-np.dot(self.A, self.x(u)))])

    def Project(self, u:np.array) -> np.array:
        return np.array([v if v >=0 else 0 for v in u])

    def Fcut1D(self, defx, x, dim):
        defx[dim] = x
        return self.F(defx)

    def Draw3DProj(self, fig, ax, vis, xdim, ydim, curX = None):
        # Trisurf variant
        # xgrid = np.arange(vis.xl, vis.xr, (vis.xr - vis.xl) * 0.05, float)
        # ygrid = np.arange(vis.yb, vis.yt, (vis.yt - vis.yb) * 0.05, float)
        # mesh = [[x, y] for y in xgrid for x in ygrid]
        #
        # gridPoints = np.array([[[x, y] for y in ygrid] for x in xgrid])
        #
        # if curX is not None:
        #     tmp = curX.copy()
        # else:
        #     tmp = self.defaultProjection.copy()
        #
        # zVals = np.array([self.Fcut2D(tmp, x, y, xdim, ydim) for x, y in mesh])
        #
        # res = ax.plot_trisurf(gridPoints[:, :, 0].flatten(), gridPoints[:, :, 1].flatten(), zVals,
        #                             cmap=cm.jet, linewidth=0, alpha=0.85)

        # # Wired surface variant
        xgrid = np.linspace(vis.xl, vis.xr, 20)
        ygrid = np.linspace(vis.yb, vis.yt, 20)
        X, Y = np.meshgrid(xgrid, ygrid)
        #Z = np.sinc(np.sqrt(X ** 2 + Y ** 2))

        if curX is not None:
            tmp = curX.copy()
        else:
            tmp = self.defaultProjection.copy()

        #Z = np.sin(np.sqrt(X ** 2 + Y ** 2  + tmp[0]))
        f = np.vectorize(lambda u,v:self.Fcut2D(tmp, u, v, xdim, ydim))
        #f = lambda u, v: u+v+1

        Z = f(X,Y)
        #print('Z: ', Z)

        res = ax.plot_wireframe(X, Y, Z)

        # self.ax.plot_surface(X, Y, Z)
        # res = self.ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
        # self.ax.plot_trisurf(x, y, zs, cmap=cm.jet, linewidth=0.2)

        # xv = [[u, w] for u, w in zip(x,y)]
        # z = [self.stableFunc(t) for t in xv]
        # self.ax.plot_wireframe(x, y, z, rstride=10, cstride=10)

        return res

    def Draw2DProj(self, fig, ax, vis, xdim, ydim, curX = None, mutableOnly = False):
        drawings = []
        x = np.arange(vis.xl, vis.xr, (vis.xr - vis.xl) * 0.01, float)
        #xv = [[t] for t in x]
        # f = np.vectorize(self.F)

        if mutableOnly:
            val = self.F(curX) if curX is not None else 0
            # F(x,y) = c1*x+c2*y => y = (F(x,y) - c1*x)/c2
            y = [(val - self.c[xdim]*t)/self.c[ydim] for t in x]
            p, = ax.plot(x, y, 'g-')
            drawings.append(p)

        if not mutableOnly:
            g = lambda x, row: (self.b[row] - self.A[row][0]*x)/self.A[row][1]
            for i in range(0, self.b.shape[0]):
                if self.A[i][1] != 0:
                    y = [g(t, i) for t in x]
                    ax.plot(x, y, 'r-')
                else:
                    ax.axvline(x=self.b[i]/self.A[i][0], ls='-', c='r')

        return drawings
