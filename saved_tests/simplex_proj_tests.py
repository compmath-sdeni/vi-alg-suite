import numpy as np

from constraints.positive_simplex_area import PositiveSimplexArea
from constraints.positive_simplex_surface import PositiveSimplexSurface
from methods.projections.simplex_proj import SimplexProj

b = 1.0
n = 2

x = np.array([-0.97, 0.99], dtype=float)
x = np.array([-0.2, 0.2], dtype=float)

n = 4
x = np.array([-2./9, 0, 0, -1./9], dtype=float)

n=6
x = np.array([-1, 1, 0, -1, 0, 2./3], dtype=float)
px = x.copy()

simpl = PositiveSimplexSurface(n, b)
xp = simpl.project(x)

#SimplexProj.doInplace(x, b)
#xp = x

x = px

print("Rand inter point: {0}".format(simpl.getSomeInteriorPoint()))

print("Projected {0} -> {1}".format(x, xp))

if not simpl.isIn(xp):
    print("Not in area!")

if abs(np.sum(xp) - b) > 0.00000001:
    print("Sum differs: {0} != {1}".format(np.sum(xp), b))

d = np.linalg.norm(x - xp)

xp[0] += 0.01
xp[1] -= 0.01
if xp[0]>=0 and xp[1]>=0 and np.linalg.norm(x - xp) - d <= 0:
    print("Dist shift 1 test failed!")
xp[0] -= 0.02
xp[1] += 0.02
if xp[0]>=0 and xp[1]>=0 and np.linalg.norm(x - xp) - d <= 0:
    print("Dist shift 2 test failed!")
