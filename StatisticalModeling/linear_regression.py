import numpy as np

arr = np.array([[0, 439.00],
                [2, 439.12],
                [4, 439.21],
                [6, 439.31],
                [8, 439.40],
                [10, 439.50]])

def find_mean(arr, col_indices):
        mean_cols = np.mean(arr[:,col_indices], axis=0)
        return mean_cols


def find_regression_line(arr):
    n = len(arr)
    column_x = arr[:, 0]
    column_y = arr[:, 1]
    x_mean = np.mean(column_x, axis=0)
    y_mean = np.mean(column_y, axis=0)
    sx =  np.std(column_x, axis=0)
    sy = np.std(column_y, axis=0)
    x_sd = (column_x - x_mean) / sx
    y_sd = (column_y- y_mean) / sy
    r = 1/n*sum(x_sd * y_sd)
    a = r*(sy/sx)
    b = y_mean - a*x_mean
    return a, b

def MSE(results):
      es = []
      for i in results:
            e = (i[1] - i[2])**2
            es.append(e)
      mse = sum(es)
      return(mse)



a, b = find_regression_line(arr)
y_pre = []
for i in arr:
    y = a*i[0] + b
    y_pre.append(y)

prediction = np.array(y_pre)
results = np.insert(arr, 2, prediction, axis=1)

mse = MSE(results)
print(f'gradient = {a} intercept = {b} and the MSE = {mse}')

