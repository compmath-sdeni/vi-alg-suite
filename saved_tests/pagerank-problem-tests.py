import numpy as np


from problems.page_rank_problem import PageRankProblem

# test from 25 bil eigenvector article. res = (0.38709677  0.12903226  0.29032258  0.19354839)
adjMatr = np.array([
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 0, 0]
])

print("Adj. matrix:\n", adjMatr)

p = PageRankProblem(adjMatr)
print("Page rank source matrix:\n", p.A)

eValues, eVectors = np.linalg.eig(p.A)

for i, v in enumerate(eValues):
    if abs(v - 1)<0.00000001:
        v = abs(eVectors[:, i])
        v /= v.sum()
        print("Eigenvector: ", v)
        print("Test: ", np.linalg.norm(p.A@v - v))
