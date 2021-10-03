import textwrap

import numpy as np
import time

from constraints.ConvexSetsIntersection import ConvexSetsIntersection
from constraints.halfspace import HalfSpace
from constraints.hyperplane import Hyperplane
from constraints.hyperplanes_bounded_set import HyperplanesBoundedSet
from constraints.hyperrectangle import Hyperrectangle
from constraints.positive_simplex_surface import PositiveSimplexSurface
from constraints.l1_sphere import L1Sphere
from constraints.l1_ball import L1Ball
from methods.projections.simplex_projection_prom import euclidean_proj_simplex, euclidean_proj_l1ball

import timeit

# b = 1.0

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
#     from constraints.b1_ball_surface import L1Sphere
#     from methods.projections.simplex_projection_prom import euclidean_proj_simplex, euclidean_proj_l1ball
#
#     n = 10000
#     b = 1
#     simpl = PositiveSimplexSurface(n, b)
#     sphere1 = L1Sphere(n, b)
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


# test_hyperplane()
# endregion

# region Hyperrectangle projection test
def test_hyperrectangle(dim: int = 10, count: int = 10):
    bounds1 = [[float(np.random.randint(-10, 10)), 0] for i in range(dim)]
    bounds = [[b[0], b[0] + float(np.random.randint(0, 10))] for b in bounds1]

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


# test_hyperrectangle(2,5)
# endregion

# region Convex sets intersection projection test
def test_convex_intersection(dim: int = 10, count: int = 10):
    # version 1
    # hr = Hyperrectangle(3, [[-5, 5], [-5, 5], [-5, 5]])
    # hp = Hyperplane(a=np.array([1., 1., 1.]), b=0.)

    hr = Hyperrectangle(5, [[-5, 5], [-5, 5], [-5, 5], [-5,5], [-5,5]])
    hp = HalfSpace(a=np.array([1., 1., 1., 1., 1.]), b=5.)

    inter = ConvexSetsIntersection([hr, hp])
    print(f"{inter}\ninterior point: {inter.getSomeInteriorPoint()}")

    # x = np.array([-20, 13, -31, 4, 15])
    # x = np.array([20, 20, 31, 21, 15])
    # x = np.array([-20, -20, -31, -21, -15])
    x = np.array([3, 0, 1, -1, 20])
    px = inter.project(x)
    print(f"{x} -> {px}; Distance: {np.linalg.norm(x - px)}; In set: {inter.isIn(px)}")


    test_failed: bool = False
    for i in range(count):
        x = np.random.random(inter.getDim()) * 20 - 10
        t = inter.project(x)
        if not inter.isIn(t):
            print(f"ERROR: {x} -> {t}")
            test_failed = True
            break
        else:
            print(f"{x} -> {t}")

    if test_failed:
        print(f"Test of {inter} failed!!! Distance to set: {inter.getDistance(t)}; Is in rect: {hr.isIn(t)}; Is in plane: {hp.isIn(t)}")
    else:
        print(f"Test of {inter} OK.")


#test_convex_intersection(2, 5)

# endregion

# region half space projection test
def test_halfspace_projection(dim: int = 10, count: int = 10):
    h = HalfSpace(a=np.array([1., 1., 1., 1., 1.]), b=5.)

    print(f"{h}\ninterior point: {h.getSomeInteriorPoint()}")

    x = np.array([20, 13, 31, 7, 3])
    px = h.project(x)
    print(f"{x} -> {px}; Distance: {np.linalg.norm(x - px)}; {np.dot(h.a, px)}")

    test_failed: bool = False
    for i in range(count):
        x = np.random.random(h.getDim()) * 10 - 5
        t = h.project(x)
        if not h.isIn(t):
            print(f"ERROR: {x} -> {t}; check: {np.dot(h.a, t)}")
            test_failed = True
            break
        else:
            print(f"{x} -> {t}; check: {np.dot(h.a, t)}")

    if test_failed:
        print(f"Test of {h} failed!!!")
    else:
        print(f"Test of {h} OK.")


# test_halfspace_projection(2, 5)
# endregion

# region L1 ball projection test
def test_L1ball_projection(dim: int = 10, count: int = 10):
    b = L1Ball(dim, b=1.)

    print(f"{b}\ninterior point: {b.getSomeInteriorPoint()}")

    # test on 2D
    i = 9.
    tb = L1Ball(2, b=1.)
    while i >= -9:
        x = np.array([i, 7.], dtype=float)
        px = tb.project(x)
        print(f"{x} -> {px}; d={np.linalg.norm(x - px)}; norm={np.sum(np.abs(px))}")
        i -= 0.5

    test_failed: bool = False
    for i in range(count):
        x = np.random.random(b.getDim()) * 3 - 1.5
        t = b.project(x)
        t_norm = np.sum(np.abs(t))
        if not b.isIn(t):
            print(f"ERROR: {x} -> {t}; check: {t_norm}")
            test_failed = True
            break
        else:
            print(f"{x} -> {t}; d={np.linalg.norm(x-t)}; norm: {t_norm}")

    if test_failed:
        print(f"Test of {b} projection failed!!!")
    else:
        print(f"Test of {b} projection OK.")

test_L1ball_projection(2, 5)
# endregion

exit()
#
# for j in range(100):
#     n = 3
#     x = np.random.random(n) * 6 - 3
#
#     simpl = PositiveSimplexSurface(n, b)
#     sphere1 = L1BallSurface(n, b)
#
#     t = x.copy()
#     xp = simpl.project(t)
#
#     xp_v2 = euclidean_proj_simplex(t)
#
#     if (np.linalg.norm(xp - xp_v2)) > 0.0000000001:
#         print("Proj differs: ", t, xp, xp_v2)
#
#     t = x.copy()
#     xp2 = sphere1.project(t)
#
#     xp2_v2 = euclidean_proj_l1ball(t)
#
#     if (np.linalg.norm(xp2 - xp2_v2)) > 0.0000000001:
#         print("Sphere1 proj differs: ", t, xp2, xp2_v2)
#
#     # SimplexProj.doInplace(x, b)
#     # xp = x
#
#     if n < 10:
#         print("Rand simplex point: {0}".format(simpl.getSomeInteriorPoint()))
#         print("Projected to simplex {0} -> {1}".format(x, xp))
#
#     tp = sphere1.getSomeInteriorPoint()
#     if n < 10:
#         print("Rand sphere1 point: {0}".format(tp))
#         print("Projected to sphere1 {0} -> {1}".format(x, xp2))
#
#     if not simpl.isIn(xp):
#         print("Projected point not in simplex area! Sum: ", xp.sum())
#
#     if not sphere1.isIn(xp2):
#         print("Projected point not not on sphere1! Sum abs: ", abs(xp2).sum())
#
#     if not sphere1.isIn(tp):
#         print("Generated sphere point not on sphere1! Sum abs: ", abs(tp).sum(), "\nPoint:\n", tp)
#
#     if abs(np.sum(xp) - b) > 0.00000001:
#         print("Simplex sum differs: {0} != {1}".format(np.sum(xp), b))
#
#     if abs(np.sum(abs(xp2)) - b) > 0.00000001:
#         print("Sphere1 sum differs: {0} != {1}".format(np.sum(abs(xp2)), b))
#
#     NTest = 10000
#     print(j, " testing...")
#
#     d = np.linalg.norm(x - xp)
#     for i in range(NTest):
#         tx = xp + (np.random.random(x.shape) * 0.02 - 0.01)
#         tx[tx < 0] = 0
#         tx /= tx.sum()
#         if (tx >= 0).all() and np.linalg.norm(x - tx) - d < 0:
#             print("Simplex proj dist shift 1 test failed! Better point: ", xp, " vs ", tx, " with sum: ", tx.sum())
#             time.sleep(5)
#             break
#
#     d = np.linalg.norm(x - xp2)
#     for i in range(NTest):
#         tx = xp2 + (np.random.random(x.shape) * 0.02 - 0.01)
#         tx /= abs(tx).sum()
#         if np.linalg.norm(x - tx) - d < 0:
#             print("Sphere1 proj dist shift 1 test failed! Better point: ", xp2, " vs ", tx, " with abs sum: ",
#                   abs(tx).sum())
#             time.sleep(5)
#             break
#
#     print("Tests finished.")
