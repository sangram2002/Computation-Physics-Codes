import numpy as np
import matplotlib.pyplot as plt

a =-2
b =1
dt =0.01
t_final =20
n_steps=int(t_final/dt)

x = 0.1
y =0.25

xs = np.zeros(n_steps)
ys = np.zeros(n_steps)

for i in range(n_steps):
    xs[i] = x
    ys[i] = y

    y += 0.5 * dt * (a * x + b * x ** 3)

    x += dt * y

    y += 0.5 * dt * (a * x + b * x ** 3)

# plot phase space
plt.plot(xs, ys)
plt.xlabel('x')
plt.ylabel('dx/dt')
plt.show()
