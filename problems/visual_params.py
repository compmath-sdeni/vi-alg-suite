class VisualParams:
    def __init__(self, *, xl=-1, xr=1, yb=-1, yt=1, zn=-1, zf=1, elev=None, azim=None, dist=None, projection:list=[0,1,2]):
        self.xl = xl
        self.xr = xr
        self.yb = yb
        self.yt = yt
        self.zn = zn
        self.zf = zf
        self.elev = elev
        self.azim = azim
        self.dist = dist
        self.projection = projection
