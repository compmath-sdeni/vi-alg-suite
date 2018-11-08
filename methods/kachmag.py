import numpy as np

A = np.array([[1, 2], [2, 1]])
b = np.array([3, 3])
x = np.array([1, 2])

A = np.array([[19, 9, 1], [9, 26, 8], [1, 8, 18]])
b = np.array([19+9+1, 9+26+8, 1+8+18])
b = np.array([40,  85,  71])
x = np.array([1, 1, 1])

r = np.linalg.norm(A@x - b)

print("A")
print(A)
print("b: ", b, "x0: ", x)

print()

j = 0
while r >= 0.001:
    j += 1
    j = j % b.shape[0]

    x = x + ((b[j] - np.dot(A[j], x)) / np.dot(A[j], A[j])) * A[j]
    print(j, A[j], b, x)

    r = np.linalg.norm(A @ x - b)

print("x: ", x, "; ||Ax-b||", r)