import textwrap

import numpy as np
import time

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.hyperplane import Hyperplane
from constraints.hyperrectangle import Hyperrectangle
from constraints.positive_simplex_surface import PositiveSimplexSurface
from constraints.b1_ball_surface import B1BallSurface
from methods.projections.simplex_projection_prom import euclidean_proj_simplex, euclidean_proj_l1ball

import timeit

b = 1.0


# n = 2
# x = np.array([-0.2, 0.3], dtype=float)

# inner point
# n = 3
# x = np.array([-0.1,0.2,-0.3], dtype="float")

# n = 4
# x = np.array([-2./9, 0, 0, -1./9], dtype=float)

# n=6
# x = np.array([-1, 1, 0, -1, 0, 2./3], dtype=float)

# timings
# setupCode = textwrap.dedent("""\
#     import numpy as np
#     from constraints.positive_simplex_surface import PositiveSimplexSurface
#     from constraints.b1_ball_surface import B1BallSurface
#     from methods.projections.simplex_projection_prom import euclidean_proj_simplex, euclidean_proj_l1ball
#
#     n = 10000
#     b = 1
#     simpl = PositiveSimplexSurface(n, b)
#     sphere1 = B1BallSurface(n, b)
#
#     x = np.random.random(n)*4 - 2
#     t = x.copy()
# """)
#
# print("Our simplex proj timing: ", timeit.timeit("xp = simpl.project(t)", setup=setupCode,  number=100))
# print("Prom simplex proj timing: ", timeit.timeit("xp_v2 = euclidean_proj_simplex(t)", setup=setupCode,  number=100))
#
# print("Our sphere proj timing: ", timeit.timeit("xp2 = sphere1.project(t)", setup=setupCode,  number=100))
# print("Prom sphere proj timing: ", timeit.timeit("xp2_v2 = euclidean_proj_l1ball(t)", setup=setupCode,  number=100))


# region Hyperplane projection test
def test_hyperplane(dim: int = 10, count: int = 10):
    h = Hyperplane(a=np.random.random(dim), b=1.51)

    test_failed: bool = False
    for i in range(count):
        x = np.random.random(h.getDim()) * 6 - 3
        t = h.project(x)
        if not h.isIn(t):
            test_failed = True
            break
        else:
            print(f"{x} -> {t} ({np.dot(t, h.a)})")

    if test_failed:
        print(f"Test of {h} failed!!!")
    else:
        print(f"Test of {h} OK.")


#test_hyperplane()
# endregion

# region Hyperrectangle projection test
def test_hyperrectangle(dim: int = 10, count: int = 10):
    bounds1 = [[float(np.random.randint(-10, 10)), 0] for i in range(dim)]
    bounds = [[b[0], b[0]+float(np.random.randint(0, 10))] for b in bounds1]

    h = Hyperrectangle(dim, bounds)

    test_failed: bool = False
    for i in range(count):
        x = np.random.random(h.getDim()) * 10 - 5
        t = h.project(x)
        if not h.isIn(t):
            print(f"ERROR: {x} -> {t}")
            test_failed = True
            break
        else:
            print(f"{x} -> {t}")

    if test_failed:
        print(f"Test of {h} failed!!!")
    else:
        print(f"Test of {h} OK.")

#test_hyperrectangle(2,5)
# endregion

# region Convex sets intersection projection test
def test_convex_intersection(dim: int = 10, count: int = 10):

    hr = Hyperrectangle(3, [[-5,5], [-5,5], [-5,5]])
    hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    inter = ConvexSetsIntersection([hr, hp])
    print(f"{inter}\ninterior point: {inter.getSomeInteriorPoint()}")

    x = np.array([-20,13,-31])
    px = inter.project(x)
    print(f"{x} -> {px}; Distance: {np.linalg.norm(x-px)}")

# [ 10 -23   3] -> [ 5. -5.  1.]; Distance: 18.788294227957003
# [-2 - 2 - 2] -> [0.33333333 0.33333333 0.33333333];  Distance: 4.041451884327381
# [-20  13 -31] -> [ 1.  5. -5.]; Distance: 34.36568055487916
    return

    test_failed: bool = False
    for i in range(count):
        x = np.random.random(h.getDim()) * 10 - 5
        t = h.project(x)
        if not h.isIn(t):
            print(f"ERROR: {x} -> {t}")
            test_failed = True
            break
        else:
            print(f"{x} -> {t}")

    if test_failed:
        print(f"Test of {h} failed!!!")
    else:
        print(f"Test of {h} OK.")


test_convex_intersection(2,5)
# endregion

exit()

for j in range(100):
    n = 3
    x = np.random.random(n) * 6 - 3

    simpl = PositiveSimplexSurface(n, b)
    sphere1 = B1BallSurface(n, b)

    t = x.copy()
    xp = simpl.project(t)

    xp_v2 = euclidean_proj_simplex(t)

    if (np.linalg.norm(xp - xp_v2)) > 0.0000000001:
        print("Proj differs: ", t, xp, xp_v2)

    t = x.copy()
    xp2 = sphere1.project(t)

    xp2_v2 = euclidean_proj_l1ball(t)

    if (np.linalg.norm(xp2 - xp2_v2)) > 0.0000000001:
        print("Sphere1 proj differs: ", t, xp2, xp2_v2)

    # SimplexProj.doInplace(x, b)
    # xp = x

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
    print(j, " testing...")

    d = np.linalg.norm(x - xp)
    for i in range(NTest):
        tx = xp + (np.random.random(x.shape) * 0.02 - 0.01)
        tx[tx < 0] = 0
        tx /= tx.sum()
        if (tx >= 0).all() and np.linalg.norm(x - tx) - d < 0:
            print("Simplex proj dist shift 1 test failed! Better point: ", xp, " vs ", tx, " with sum: ", tx.sum())
            time.sleep(5)
            break

    d = np.linalg.norm(x - xp2)
    for i in range(NTest):
        tx = xp2 + (np.random.random(x.shape) * 0.02 - 0.01)
        tx /= abs(tx).sum()
        if np.linalg.norm(x - tx) - d < 0:
            print("Sphere1 proj dist shift 1 test failed! Better point: ", xp2, " vs ", tx, " with abs sum: ",
                  abs(tx).sum())
            time.sleep(5)
            break

    print("Tests finished.")
