import numpy as np

def squer_matrix(size, *elemnts):
    if len(elemnts) != size**2:
        raise ValueError('The number of elements does not match the size of the matrix')
    return np.array(elemnts).reshape(size, size)

matrix = squer_matrix(3, -26, 33, -25, 31, 42, 23, -11, -15, -4)
print(matrix)

def eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    for i in range(len(eigenvalues)):
        # Перевірка рівності A⋅v=λ⋅v
        if not np.allclose(np.dot(matrix, eigenvectors[:, i]), eigenvalues[i] * eigenvectors[:, i]):
            return False, "A⋅v != λ⋅v for eigenvalue {} and its corresponding eigenvector".format(i)

    return eigenvalues, eigenvectors


values, vectors = eigen(matrix)
print("values:")
print(values)
print("vectors:")
print(vectors)

