import numpy as np
n = 5
matrix = np.zeros([n, n])

for i in range(int(n / 2)):
    length = n - i * 2 - 1
    before = 4 * (n - i) * i
    matrix[i, i: i + length] = [x + before + 1 for x in range(length)]
    matrix[i: i + length, i + length] = [x + before + length + 1 for x in range(length)]
    matrix[i + length, i + 1: i + length + 1] = [-x + before + length * 3 for x in range(length)]
    matrix[i + 1: i + length + 1, i] = [-x + before + length * 4 for x in range(length)]

if n % 2 == 1:
    idx = int(n/2)
    matrix[idx, idx] = n ** 2

print(matrix)