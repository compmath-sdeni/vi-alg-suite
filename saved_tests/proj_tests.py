import numpy as np

from constraints.positive_simplex_area import PositiveSimplexArea
from constraints.positive_simplex_surface import PositiveSimplexSurface
from constraints.b1_ball_surface import B1BallSurface
from methods.projections.simplex_proj import SimplexProj

b = 1.0
n = 2

x = np.array([-0.2, 0.3], dtype=float)

# n = 4
# x = np.array([-2./9, 0, 0, -1./9], dtype=float)

# n=6
# x = np.array([-1, 1, 0, -1, 0, 2./3], dtype=float)

n = 10000
x = np.random.random(n)*4 - 2

simpl = PositiveSimplexSurface(n, b)
t = x.copy()
xp = simpl.project(t)

sphere1 = B1BallSurface(n, b)
t = x.copy()
xp2 = sphere1.project(t)

#SimplexProj.doInplace(x, b)
#xp = x

if n < 10:
    print("Rand simplex point: {0}".format(simpl.getSomeInteriorPoint()))
    print("Projected to simplex {0} -> {1}".format(x, xp))

tp = sphere1.getSomeInteriorPoint()
if n < 10:
    print("Rand sphere1 point: {0}".format(tp))
    print("Projected to sphere1 {0} -> {1}".format(x, xp2))

if not simpl.isIn(xp):
    print("Projected point not in simplex area! Sum: ", xp.sum())

if not sphere1.isIn(xp2):
    print("Projected point not not on sphere1! Sum abs: ", abs(xp2).sum())

if not sphere1.isIn(tp):
    print("Generated sphere point not on sphere1! Sum abs: ", abs(tp).sum(), "\nPoint:\n", tp)

if abs(np.sum(xp) - b) > 0.00000001:
    print("Simplex sum differs: {0} != {1}".format(np.sum(xp), b))

if abs(np.sum(abs(xp2)) - b) > 0.00000001:
    print("Sphere1 sum differs: {0} != {1}".format(np.sum(abs(xp2)), b))

NTest = 10000
print("Testing...")

d = np.linalg.norm(x - xp)
for i in range(NTest):
    tx = xp + (np.random.random(x.shape) * 0.02 - 0.01)
    tx[tx<0] = 0
    tx /= tx.sum()
    if (tx >= 0).all() and np.linalg.norm(x - tx) - d <= 0:
        print("Simplex proj dist shift 1 test failed! Better point: ", tx, " with sum: ", tx.sum())
        break

d = np.linalg.norm(x - xp2)
for i in range(NTest):
    tx = xp2 + (np.random.random(x.shape) * 0.02 - 0.01)
    tx /= abs(tx).sum()
    if np.linalg.norm(x - tx) - d <= 0:
        print("Sphere1 proj dist shift 1 test failed! Better point: ", tx, " with abs sum: ", abs(tx).sum())
        break