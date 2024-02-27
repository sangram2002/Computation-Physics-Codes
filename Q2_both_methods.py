import numpy as np
import matplotlib.pyplot as plt

a =1
b =-2
x0 =0.1
y0 =0.25
t_final =20
dt =0.01
n_steps=int(t_final / dt)

x_values_euler =np.zeros(n_steps)
y_values_euler =np.zeros(n_steps)
x_verlet_method =np.zeros(n_steps)
y_verlet_method =np.zeros(n_steps)
energies_euler_method =np.zeros(n_steps)
energies_verlet_method =np.zeros(n_steps)

# explicit Euler method
x_values_euler[0] = x0
y_values_euler[0] = y0
for i in range(1, n_steps):
    energies_euler_method[i - 1] =0.5 *y_values_euler[i-1]**2+0.5*a*x_values_euler[i - 1]**2

    x_values_euler[i] = x_values_euler[i - 1]+y_values_euler[i - 1]*dt
    y_values_euler[i] = y_values_euler[i - 1] +(a * x_values_euler[i - 1] +b*x_values_euler[i - 1] ** 3)*dt
    # print(x_values_euler[i])
    # print(y_values_euler[i])


# Verlet method
x_verlet_method[0] = x0
y_verlet_method[0] = y0
for i in range(1, n_steps):
    energies_verlet_method[i - 1] = 0.5 * y_verlet_method[i - 1]**2+0.5*a*x_verlet_method[i - 1]** 2

    y_verlet_method[i - 1] += 0.5 * dt*(a*x_verlet_method[i - 1]+b*x_verlet_method[i - 1]**3)

    x_verlet_method[i] = x_verlet_method[i - 1]+dt*y_verlet_method[i - 1]

    y_verlet_method[i] = y_verlet_method[i - 1]+0.5*dt*(a*x_verlet_method[i]+b*x_verlet_method[i]**3)
    # print('Hello')
    # print(x_verlet_method[i])

# plot phase space for both methods
fig, ax = plt.subplots()
ax.plot(x_values_euler, y_values_euler, label='Euler')
ax.plot(x_verlet_method, y_verlet_method, label='Verlet')
ax.set_xlabel('Position')
ax.set_ylabel('Velocity')
ax.set_title('Phase Space')
ax.legend()

plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\Q2_phase_space.png')

# plot total energy
fig, ax = plt.subplots()
t = np.arange(0, t_final, dt)
ax.plot(t, energies_euler_method, label='Euler_method')
ax.plot(t, energies_verlet_method, label='Verlet_method')
ax.set_xlabel('Time')
ax.set_ylabel('Total energy')
ax.set_title('Total Energy vs Time')
ax.legend()

# plt.show()
plt.savefig('D:\MY_NOTES\CP_bhoosan\Codes_py\Q2_total_energy.png')

