import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class Parameters(object):
    pass


def control(t, u_param):
    u = u_param.C1 + u_param.C2 * np.exp(t)
    return u


def rhs(t, x, u_param):
    dxdt = np.zeros(2)

    x1 = x[0]
    x2 = x[1]

    u = control(t, u_param)

    dxdt[0] = x2
    dxdt[1] = - x2 + u

    return dxdt


def get_C1C2(u_param, x1f, x2f, tf):
    exp_T = np.exp(tf)
    A = x2f + x1f
    B = (x2f - x1f) * exp_T
    u_param.C1 = (A - B) / (tf + 2 + (tf - 2) * exp_T)
    u_param.C2 = (A - u_param.C1 * tf) / (exp_T - 1)


t0 = 0          # start time
tf = 2          # final time
dt = 0.002      # step-size
N = 1000

u_param = Parameters()

x2_range = np.linspace(0, 3, N + 1)
x1_range = 15 - 5 * x2_range
J = np.zeros(N + 1)
it = 0

for x2 in x2_range:
    x2f = x2
    x1f = 15 - 5 * x2f

    get_C1C2(u_param, x1f, x2f, tf)   
    exp_T = np.exp(tf)
    exp_2T = np.exp(2 * tf)
 
    u_integ = u_param.C1**2 * tf + 2 * u_param.C1 * u_param.C2 * (exp_T - 1) + u_param.C2**2 * (exp_2T - 1) / 2
    J_step = ((x1f - 5)**2 + (x2f - 2)**2 + u_integ) / 2
    
    J[it] = J_step
    it += 1


x2_best = x2_range[np.argmin(J)]
x1_best = 15 - 5 * x2_best
u_param_best = Parameters()
u_param_best.x1f = x1_best
u_param_best.x2f = x2_best
get_C1C2(u_param_best, x1_best, x2_best, tf)

tt = np.arange(t0, tf, dt)
x0 = [0, 0]

sol = sci.solve_ivp(lambda t, x: rhs(t, x, u_param_best), (t0, tf), x0, t_eval=tt)
y_traj = sol.y
# print(y_traj)
# plt.plot(tt, y_traj)
plt.subplot(2, 1, 1)
plt.plot(y_traj[0], y_traj[1])
plt.plot(x1_range, x2_range)
plt.vlines(x1_best, ymin=0, ymax=np.max(x2_range), colors="r", linestyles="dashdot")
plt.title("best trajectory")
plt.legend(["trajectory", "final line"])
plt.grid(ls=":")
plt.xlabel("x1")
plt.ylabel("x2")

plt.subplot(2, 1, 2)
plt.plot(x1_range, J)
plt.vlines(x1_best, ymin=0, ymax=np.max(J), colors="r", linestyles="dashdot")
plt.title("functional")
plt.xlabel("x1")
plt.grid(ls=":")
# plt.legend(['x1', 'x2'])
plt.show()