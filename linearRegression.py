import numpy as np
import time

true_vector = np.array([1.0,2.0]) # Vector that we want to learn
d = len(true_vector)

points = []

for i in range(10000):
    x = np.random.rand(d)
    y = true_vector.dot(x) + np.random.rand() # add a litle bit noise
    points.append((x,y))

def F(w):
    return sum((w.dot(x) - y) ** 2 for x,y in points) / len(points)

def dF(w):
    return sum( 2.0 * (w.dot(x) - y) * x for x,y in points) / len(points)

def sF(w, i):
    x, y = points[i]
    return (w.dot(x) - y) ** 2

def sdF(w, i):
    x, y = points[i]
    return 2 * (w.dot(x) - y) * x

def gradientDescent(F, dF, d):
    w = np.zeros(d)
    eta = 0.01
    value = 0

    for t in range(1000):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {} , F(w) = {}'.format(t,w,value))

def stochasticGradientDescent(sF, sdF, d, n):
    numUpdates = 0
    w = np.zeros(d)
    value = 0
    eta = 0.01

    start = time.time() # start timer

    for t in range(1000):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numUpdates += 1
            eta = 1.0 / numUpdates
            w = w - eta * gradient
        # print('iteration {}: w = {} , F(w) = {}'.format(t,w,value))

    stop = time.time() # stop timer
    
    print('w = {}, F(w) = {}, time = {}'.format(w, value, (stop-start)))


def normalEquation():
    X = []
    Y = []
    # changing input and label types to normal 
    for a,b in points:
        X.append(a) 
        Y.append(b)
    
    X = np.array(X)
    Y = np.array(Y)


    start = time.time() # start timer

    Xt = np.transpose(X)
    w = np.linalg.inv( Xt.dot(X) ).dot(Xt.dot(Y))

    stop = time.time() # stop timer

    print('w = {}, F(w) = {}, time = {}'.format(w, F(w), (stop-start)))


#gradientDescent(F, dF, d)
normalEquation()
stochasticGradientDescent(sF, sdF, d, len(points))
