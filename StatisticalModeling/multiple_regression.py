import numpy as np
import numpy as np




def matrix_multiplication(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result

def print_matrix(matrix):
    for row in matrix:
        print(row)


# a funtion to do the transpose of a matrix
def matrix_transpose(matrix):
    result = np.transpose(matrix)
    return result

# a function to find the inverse of a matrix
def matrix_inverse(matrix):
    result = np.linalg.inv(matrix)
    return result

# a function to stardise a matrix
def standardize_matrix(matrix):
    result = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    return result

# a function to revese the standardisation of a matrix
def reverse_standardize_matrix(matrix):
    result = (matrix * np.std(matrix, axis=0)) + np.mean(matrix, axis=0)
    return result

def predict(X_test, beta, residuals):
    return matrix_multiplication(X_test, beta) + residuals.shape[0]



X = np.array([[1500, 3],
              [2000, 4],
              [1800, 3],
              [2200, 4],
              [1600, 3]])

Y = np.array([[300000],
              [400000],
              [350000],
              [450000],
              [320000]])

# stardise the data

X = standardize_matrix(X)
Y = standardize_matrix(Y)

#add b0 the intercept
X = np.insert(X, 0, 1, axis=1)

#find the transpose of X 
X_t = matrix_transpose(X)

# multiply X_t and X
argument_1 = matrix_inverse(matrix_multiplication(X_t, X))
argument_2 = matrix_multiplication(X_t, Y)

beta = matrix_multiplication(argument_1, argument_2)

resduals = Y - matrix_multiplication(X, beta)

X_test = np.array([[1, 1500, 3],
                    [1, 2000, 4],
                    [1, 1800, 3],
                    [1, 2200, 4],
                    [1, 1600, 3]])

Y_test = predict(X_test, beta, resduals)

results = reverse_standardize_matrix(Y_test)
print_matrix(results)


