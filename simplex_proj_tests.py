import numpy as np

from constraints.positive_simplex_area import PositiveSimplexArea
from methods.projections.simplex_proj import SimplexProj

b = 4.0

x = np.array([1, -2], dtype=float)
px = x.copy()

simpl = PositiveSimplexArea(2, b)
# xp = simpl.project(x)

SimplexProj.doInplace(x, b)
xp = x
x = px

print("{0}".format(simpl.getSomeInteriorPoint()))

print("{0} -> {1}".format(x, xp))

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
