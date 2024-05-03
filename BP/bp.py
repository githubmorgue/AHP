from numpy import array, exp, random, dot

x = array()
y = array().T
random.seed(1)
weights = 2 * random.random((3, 1)) - 1

for it in range(10000):
    output = 1/(1+exp(-dot(x, weights)))
    error = y - output
    delta = error * output * (1-output)
    weights += dot(x.T, delta)

print(weights)
print(1/(1+exp(-dot([[1,0,0]], weights))))