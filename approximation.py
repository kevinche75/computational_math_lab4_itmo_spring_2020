import numpy as np
from math import e
import matplotlib.pyplot as plt

def calc_linear_coefs(x, y):
    SX = np.sum(x)
    SXX = np.sum(x**2)
    SY = np.sum(y)
    SXY = np.sum(x*y)
    n = len(x)
    denominator = SXX*n - SX*SX
    assert  denominator != 0, "Cann't calculate coefs: divizion by zero"
    a = (SXY*n - SX*SY)/denominator
    b = (SXX*SY-SX*SXY)/denominator
    return [a,b]

def calc_quadratic_coefs(x, y):
    X_1 = np.sum(x)
    X_2 = np.sum(x**2)
    X_3 = np.sum(x**3)
    X_4 = np.sum(x**4)
    Z_1 = np.sum(y)
    Z_2 = np.sum(x*y)
    Z_3 = np.sum(y*x**2)
    n = len(x)
    delta =   np.linalg.det([[  n, X_1, X_2],
                             [X_1, X_2, X_3],
                             [X_2, X_3, X_4]])
    assert delta != 0, "Cann't calculate coefs: divizion by zero"
    delta_0 = np.linalg.det([[Z_1, X_1, X_2],
                             [Z_2, X_2, X_3],
                             [Z_3, X_3, X_4]])
    delta_1 = np.linalg.det([[  n, Z_1, X_2],
                             [X_1, Z_2, X_3],
                             [X_2, Z_3, X_4]])
    delta_2 = np.linalg.det([[  n, X_1, Z_1],
                             [X_1, X_2, Z_2],
                             [X_2, X_3, Z_3]])

    return [delta_0/delta, delta_1/delta, delta_2/delta]

def calc_exp_coefs(x, y):
    assert not np.any(y <= 0), "Cann't calculate coefs, y <= 0"
    a_1, a_0 = calc_linear_coefs(x, np.log(y))
    return [e**a_0, a_1]
    
def calc_pow_coefs(x, y):
    assert not (np.any(x <= 0) or np.any(y <= 0)), "Cann't calulcate coefs, x <= 0 or y <= 0"
    a_1, a_0 = calc_linear_coefs(np.log(x), np.log(y))
    return [e**a_0, a_1]

def calc_log_coefs(x,y):
    assert not np.any(x <= 0), "Cann't calculate coefs, x <= 0"
    a_0, a_1 = calc_linear_coefs(np.log(x), y)
    return [a_0, a_1]

def exclude_noise(func, x, y):
    index = np.argmax((func(x)-y)**2)
    print(x[index])
    print(y[index])
    return np.delete(x, index), np.delete(y, index)

class Function(object):

    def __init__(self, f_type):

        self.coefs = []
        self.recalc_coefs = []

        if f_type == 0:
            self.function = lambda a, b, x: a*x + b
            self.description = "{0:.3f}x + {1:.3f}"
        elif f_type == 1:
            self.function = lambda a_0, a_1, a_2, x: a_2*x**2 + a_1*x + a_0
            self.description = "{0:.3f} + {1:.3f}x + {2:.3f}x^2"
        elif f_type == 2:
            self.function = lambda a, b, x: a*e**(b*x)
            self.description = "{0:.3f}e^({1:.3f}x)"
        elif f_type == 3:
            self.function = lambda a, b, x: a*np.log(x)+b
            self.description = "{0:.3f}log(x) + {1:.3f}"
        else:
            self.function = lambda a,b, x: a*x**b
            self.description = "{0:.3f}x^{1:.3f}"

    def __call__(self, x, recalc_use = False):
        if recalc_use:
            coefs = self.recalc_coefs
        else:
            coefs = self.coefs
        return self.function(*coefs, x)

    def get_description(self, recalc_use = False):
        if recalc_use:
            return self.description.format(*self.recalc_coefs)
        else:
            return self.description.format(*self.coefs)

class LeastSquares(object):

    def __init__(self, f_type, x, y):
        self.x = x  
        self.y = y
        self.function = Function(f_type)
        if f_type == 0:
            self.f_coefs = calc_linear_coefs
        elif f_type == 1:
            self.f_coefs = calc_quadratic_coefs
        elif f_type == 2:
            self.f_coefs = calc_exp_coefs
        elif f_type == 3:
            self.f_coefs = calc_log_coefs
        else:
            self.f_coefs = calc_pow_coefs

    def calc_coefs(self):
        coefs = self.f_coefs(self.x, self.y)
        self.function.coefs = coefs

    def recalc_coefs(self):
        x, y = exclude_noise(self.function, self.x, self.y)
        coefs = self.f_coefs(x, y)
        self.function.recalc_coefs = coefs

    def draw_graphics(self):
        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(self.x, self.y, label = "source", c = "r")
        b = max(self.x)
        a = min(self.x)
        step = (b-a)/10000
        x = np.arange(a, b, step)
        cells = [[self.function.coefs]]
        row_labels = ["before exclude"]
        ax.plot(x, self.function(x), label = "approximation before exclude: {0}".format(self.function.get_description()))
        if len(self.function.recalc_coefs)!= 0:
            ax.plot(x, self.function(x, recalc_use=True), label = "approximation after exclude: {0}".format(self.function.get_description(True)))
            cells.append([self.function.recalc_coefs])
            row_labels.append("after exclude")
        ax.table(cellText = cells, rowLabels = row_labels, colLabels = [""])
        ax.legend()
        plt.show()