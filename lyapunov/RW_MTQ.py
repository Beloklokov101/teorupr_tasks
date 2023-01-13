import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp 


# Вспомогательные функции

def normalize(obj):

    return obj / np.linalg.norm(obj)


def cross_product(a, b):

    def check_dimensions(vec, string):

        if vec.ndim != 1:
            raise Exception("The {} input is not a vector".format(string))
        if len(vec) != 3:
            raise Exception("Wrong number of coordinates in the {0} vector: {1}, should be 3".format(string, len(vec)))

    check_dimensions(a, 'first')
    check_dimensions(b, 'second')

    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

def quat_product(q1, q2):

    def check_dimensions(q, string):

        if q.ndim != 1:
            raise Exception("The {} input is not a quaternion".format(string))
        if len(q) != 4:
            raise Exception("Wrong number of coordinates in the {0} quaternion: {1}, should be 4".format(string, len(q)))

    check_dimensions(q1, 'first')
    check_dimensions(q2, 'second')

    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - q1[1:].dot(q2[1:])
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + cross_product(q1[1:], q2[1:])

    return q

def rotate_vec_with_quat(q, vec):

    def check_dimensions(obj, is_quat):

        if obj.ndim != 1:
            raise Exception("Not a {}".format('quaternion' * is_quat + 'vector' * (1 - is_quat)))
        if len(obj) != (3 + 1 * is_quat):
            raise Exception("Wrong number of coordinates in the {0}: {1}, should be {2}"
                            .format('quaternion' * is_quat + 'vector' * (1 - is_quat), len(obj), 3 + 1 * is_quat))

    check_dimensions(q, True)
    check_dimensions(vec, False)

    q = quat_conjugate(q)

    qxvec = cross_product(q[1:], vec)

    return q[1:].dot(vec) * q[1:] + q[0]**2. * vec + 2. * q[0] * qxvec + cross_product(q[1:], qxvec)

def quat2rpy(q0, q1, q2, q3):

    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1**2 + q2**2))
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2**2 + q3**2))

    return [roll, pitch, yaw]

def quat2rpy_deg(q0, q1, q2, q3):
    
    norm_q = np.linalg.norm([q0, q1, q2, q3]) 
    # norm_q = np.linalg.norm([q0, q1, q2, q3], axis=0) 
    q0, q1, q2, q3 = q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q 
    

    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1**2 + q2**2))*180/np.pi
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))*180/np.pi
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2**2 + q3**2))*180/np.pi

    return [roll, pitch, yaw]

def quat_conjugate(q):

    q_new = np.copy(q)
    q_new[1:] *= -1.

    return q_new


MU_0 = 1.257e-6                                # [N / A^2] vacuum permeability
MU_e = 7.94e+22                                # [A * m^2] magnetic dipole moment of the Earth

R_e = 6371e+3                                  # [m] radius of the Earth
# altitude = 450e+3                              # [m] altitude of the orbit
altitude = 409e+3                              # [m] altitude of the orbit
# incl = 87*np.pi / 180                       # [rad] inclination of the orbit
incl = 51.63*np.pi / 180                       # [rad] inclination of the orbit

R_orb = R_e + altitude                         # [m] radius of the orbit
B_0 = MU_e * MU_0 / (4 * np.pi * R_orb**3.)    # [T] magnitude of the magnetic field  on the orbit

def B_fun(u, i, B_0):

    """
    Magnetic field on the orbit
    :param u: latitude in rad
    :param i: inclination in rad
    :returns: 3-axis magnetic field
    """

    return np.array([np.cos(u)*np.sin(i), np.cos(i), -2.*np.sin(u)*np.sin(i)]) * B_0


def ctrl_torque(t, quat, omega, h, ctrl, J):
    
    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    dw = omega - rotate_vec_with_quat(dq, ctrl.omega_req)

    M_ctrl = np.zeros(3)
    M_ctrl = cross_product(omega, J @ omega + h) - ctrl.K_d @ dw - dq[0] * ctrl.K_p @ dq[1:]
    
    # if np.max(np.abs(M_ctrl)) > ctrl.trq_max:
    #     M_ctrl *= ctrl.trq_max / np.max(np.abs(M_ctrl))  
    
    # for i in range(3):
    #     if np.abs(h[i]) / ctrl.h_max > 0.99 and M_ctrl[i] * h[i] < 0:
    #         M_ctrl[i] = 0
    
    u_angle = ctrl.w0 * t
    B_o = B_fun(u_angle, incl, B_0)
    B_b = rotate_vec_with_quat(dq, B_o)

    # print(ctrl.S, B_b)
    v = ctrl.v
    k = ctrl.S[np.argmin(np.abs(B_b[ctrl.S]))]
    l = ctrl.S[np.argmax(np.abs(B_b[ctrl.S]))]

    lambda_l = M_ctrl[v] / B_b[l]
    if np.abs(lambda_l) > ctrl.m_max:
        lambda_l *= ctrl.m_max / np.abs(lambda_l)

    u, m = np.zeros(3), np.zeros(3)
    u[k] = M_ctrl[k]
    u[l] = M_ctrl[l] + B_b[v] * lambda_l
    m[k] = lambda_l
    if np.max(np.abs(u)) > ctrl.dh_max:
        u *= ctrl.dh_max / np.max(np.abs(u))

    for i in range(3):
        if np.abs(h[i]) / ctrl.h_max > 0.99 and u[i] * h[i] < 0:
            u[i] = 0
    
    M_ctrl = u + cross_product(m, B_b)
    
    return u, M_ctrl, B_b, B_o, lambda_l


def ctrl_torque_APF(t, quat, omega, h, ctrl, J):
    
    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    dw = omega - rotate_vec_with_quat(dq, ctrl.omega_req)

    # M_ctrl = np.zeros(3)
    # M_ctrl = cross_product(omega, J @ omega + h) - ctrl.K_d @ dw - dq[0] * ctrl.K_p @ dq[1:]
    
    u_angle = ctrl.w0 * t
    B_o = B_fun(u_angle, incl, B_0)
    B_b = rotate_vec_with_quat(dq, B_o)

    l_vec = np.array([np.sin(u_angle), 0, 2 * np.cos(u_angle)])
    dBo_i = - 3 * ctrl.w0 * B_0 * np.sin(incl) * l_vec
    dBb_i = rotate_vec_with_quat(dq, dBo_i)
    dBb_b = dBb_i - cross_product(omega, B_b)
    
    psi = ctrl.alpha * ctrl.gamma * np.exp(-ctrl.gamma * (np.sum(B_b[ctrl.S] ** 2)))
    p = np.sum(B_b[ctrl.S] * dBb_b[ctrl.S]) * dw / np.sum(dw ** 2)

    M_ctrl = np.zeros(3)
    M_ctrl = cross_product(omega, J @ omega + h) - ctrl.K_d @ dw - ctrl.K_p @ (dq[0] * dq[1:] + psi * p)


    # print(ctrl.S, B_b)
    v = ctrl.v
    k = ctrl.S[np.argmin(np.abs(B_b[ctrl.S]))]
    l = ctrl.S[np.argmax(np.abs(B_b[ctrl.S]))]

    lambda_l = M_ctrl[v] / B_b[l]
    if np.abs(lambda_l) > ctrl.m_max:
        lambda_l *= ctrl.m_max / np.abs(lambda_l)

    u, m = np.zeros(3), np.zeros(3)
    u[k] = M_ctrl[k]
    u[l] = M_ctrl[l] + B_b[v] * lambda_l
    m[k] = lambda_l
    if np.max(np.abs(u)) > ctrl.dh_max:
        u *= ctrl.dh_max / np.max(np.abs(u))

    for i in range(3):
        if np.abs(h[i]) / ctrl.h_max > 0.99 and u[i] * h[i] < 0:
            u[i] = 0
    
    M_ctrl = u + cross_product(m, B_b)
    
    return u, m, M_ctrl



def rhs(t, x, sat, ctrl):

    # quat = x[:4] / np.linalg.norm(x[:4])
    # omega = x[4:7]
    # h = x[7:10]
     
    # omega_rel = omega - rotate_vec_with_quat(quat, np.array([0., sat.mean_motion, 0.]))    
        
    # trq_gg = gravity_gradient_torque(quat, sat.J, sat.mean_motion)
    # trq_dst = np.random.normal(0, sat.dst_sigma)
    # action = ctrl_torque(ctrl, sat.J_ctrl, omega, quat, h, trq_gg)        

    # x_dot = np.zeros(10)

    # x_dot[0] = -0.5 * quat[1:].dot(omega_rel)
    # x_dot[1:4] = 0.5 * (quat[0] * omega_rel + cross_product(quat[1:], omega_rel))
    # x_dot[4:7] = sat.J_inv.dot(trq_gg + trq_dst + action - cross_product(omega, sat.J.dot(omega) + h))
    # x_dot[7:10] = - action
    
    # return x_dot

    quat = x[:4] / np.linalg.norm(x[:4])
    omega = x[4:7]
    h = x[7:]

    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    dw = omega - rotate_vec_with_quat(dq, ctrl.omega_req)

    u, M_ctrl, B_b, B_o, lambda_l = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(1)
    u, M_ctrl, B_b, B_o, lambda_l = ctrl_torque(t, quat, omega, h, ctrl, sat.J)

    x_dot = np.zeros(10)

    x_dot[0] = -0.5 * quat[1:] @ dw
    x_dot[1:4] = 0.5 * (quat[0] * dw + cross_product(quat[1:], dw))
    x_dot[4:7] = sat.J_inv @ (M_ctrl - cross_product(omega, sat.J @ omega + h))
    x_dot[7:] = -u
    
    # print(t)
    return x_dot




class Parameters(object):
    pass

sat = Parameters()
sat.J = np.diag(np.array([0.0309, 0.0319, 0.0051]))
sat.J_inv = np.linalg.inv(sat.J)

# a = 0.32
ctrl = Parameters()

ctrl.v = 1
indexes = np.array([0, 1, 2])
ctrl.S = np.delete(indexes, np.where(indexes == ctrl.v))
K = np.eye(3)
# K[ctrl.v, ctrl.v] = 0.05
ctrl.K_p = 5e-3 * K
# ctrl.K_p = 5e-1 * K
ctrl.K_d = 1e-2 * K
# ctrl.K_d = 1e-3 * K
ctrl.dh_max = 2e-3
ctrl.h_max = 19e-3
ctrl.m_max = 100

ctrl.omega_req = np.zeros(3)
# ctrl.w0 = 360 * np.pi / 180 / (24 * 60 * 60)
ctrl.w0 = 0.02
ctrl.omega_req[1] = ctrl.w0
# ctrl.omega_req = np.array([0., 0.5, 0.])
ctrl.q_req = np.array([1., 0., 0., 0.])
# ctrl.q_req = np.array([np.cos(np.pi/3), 0.1, np.sin(np.pi/6), 0])

alpha_0 = np.pi / 4
# x_0 = np.array([np.cos(alpha_0), 0, np.sin(alpha_0), 0, 0, 0, 0, 0, 0, 0])
x_0 = np.array([np.cos(alpha_0), 0, 0, np.sin(alpha_0), 0, 0, 0, 0, 0, 0])
# x_0 = np.array([1., 0., 0., 0, 0, 0, 0, 0, 0, 0])

t0 = 0.
tf = 100.
t_eval = np.arange(t0, tf, 0.01)

sol = solve_ivp(lambda t, x: rhs(t, x, sat, ctrl), (t0,tf), x_0, t_eval = t_eval)
x = sol.y.T
t = sol.t


roll, pitch, yaw = quat2rpy_deg(x[1:,0], x[1:,1], x[1:,2], x[1:,3])

u = np.zeros((np.size(t), 3))
M_ctrl_final, B_b, B_o = np.zeros((np.size(t), 3)), np.zeros((np.size(t), 3)), np.zeros((np.size(t), 3))
lambda_l = np.zeros(np.size(t))
B_b = np.zeros((np.size(t), 3))
B_o = np.zeros((np.size(t), 3))
k = np.zeros(np.size(t))
l = np.zeros(np.size(t))
dw = np.zeros((np.size(t), 3))
M_ctrl_first = np.zeros((np.size(t), 3))
for i in range(np.size(t)):
    quat = x[i,:4] / np.linalg.norm(x[i,:4])
    omega = x[i,4:7]
    h = x[i,7:]
    u[i], M_ctrl_final[i], B_b[i], B_o[i], lambda_l[i] = ctrl_torque(t[i], quat, omega, h, ctrl, sat.J)
    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    dw[i] = omega - rotate_vec_with_quat(dq, ctrl.omega_req)
    M_ctrl_first[i] = cross_product(omega, sat.J @ omega + h) - ctrl.K_d @ dw[i] - dq[0] * ctrl.K_p @ dq[1:]
    k[i] = ctrl.S[np.argmin(np.abs(B_b[i, ctrl.S]))]
    l[i] = ctrl.S[np.argmax(np.abs(B_b[i, ctrl.S]))]


#     B_b = np.zeros((np.size(t), 3))
#     B_o = np.zeros((np.size(t), 3))
#     k = np.zeros(np.size(t))
#     l = np.zeros(np.size(t))
#     dw = np.zeros((np.size(t), 3))
#     M_ctrl_first = np.zeros((np.size(t), 3))
#     for i in range(np.size(t)):
#         quat = x[i,:4] / np.linalg.norm(x[i,:4])
#         omega = x[i, 4:7]
#         dq = quat_product(quat_conjugate(ctrl.q_req), quat)
#         dw[i] = omega - rotate_vec_with_quat(dq, ctrl.omega_req)
#         M_ctrl_first[i] = cross_product(omega, sat.J @ omega + x[i, 7:]) - ctrl.K_d @ dw[i] - dq[0] * ctrl.K_p @ dq[1:]
#         u_angle = ctrl.w0 * t[i]
#         B_o[i] = B_fun(u_angle, incl, B_0)
#         B_b[i] = rotate_vec_with_quat(dq, B_o[i])
#         k[i] = ctrl.S[np.argmin(np.abs(B_b[i, ctrl.S]))]
#         l[i] = ctrl.S[np.argmax(np.abs(B_b[i, ctrl.S]))]


roll, pitch, yaw = quat2rpy_deg(x[:,0], x[:,1], x[:,2], x[:,3])

fig1 = plt.figure(figsize=(16,12))
ax1 = fig1.add_subplot(3,2,1)

ax1.set_title("Euler Angles")
ax1.plot(t, roll, label = 'roll', color = 'red')
ax1.plot(t, pitch, label = 'pitch', color = 'green')
ax1.plot(t, yaw, label = 'yaw', color = 'blue')
ax1.set_ylabel(r'angles, [deg]')
ax1.set_xlabel(r't, [s]')
ax1.grid(True)
ax1.legend()

ax2 = fig1.add_subplot(3,2,2)

ax2.set_title("B_b")
ax2.plot(t, B_b[:, 0], label = '$Bb_x$', color = 'red')
ax2.plot(t, B_b[:, 1], label = '$Bb_y$', color = 'green')
ax2.plot(t, B_b[:, 2], label = '$Bb_z$', color = 'blue')
ax2.set_ylabel(r'B_b')
ax2.set_xlabel(r't, [s]')
ax2.grid(True)
ax2.legend()

ax3 = fig1.add_subplot(3,2,3)

ax3.set_title("RW angular momentum")
ax3.plot(t, x[:,7], label = '$h_x$', color = 'red')
ax3.plot(t, x[:,8], label = '$h_y$', color = 'green')
ax3.plot(t, x[:,9], label = '$h_z$', color = 'blue')
ax3.set_ylabel(r'h, [Nms]')
ax3.set_xlabel(r't, [s]')
ax3.grid(True)
ax3.legend()

#     ax3.set_title("lambda_l")
#     ax3.plot(t, lambda_l, label = '$\lambda_l$', color = 'red')
#     ax3.set_ylabel(r'lambda_l')
#     ax3.set_xlabel(r't, [s]')
#     ax3.grid(True)
#     ax3.legend()

ax4 = fig1.add_subplot(3,2,4)

ax4.set_title("M_ctrl_first")
#     ax4.plot(t, M_ctrl_final[:, 0] - M_ctrl_first[:, 0], label = '$Mctrl_x$', color = 'red')
#     ax4.plot(t, M_ctrl_final[:, 1] - M_ctrl_first[:, 1], label = '$Mctrl_y$', color = 'green')
#     ax4.plot(t, M_ctrl_final[:, 2] - M_ctrl_first[:, 2], label = '$Mctrl_z$', color = 'blue')
ax4.plot(t, M_ctrl_first[:, 0], label = '$Mctrl_x$', color = 'red')
ax4.plot(t, M_ctrl_first[:, 1], label = '$Mctrl_y$', color = 'green')
ax4.plot(t, M_ctrl_first[:, 2], label = '$Mctrl_z$', color = 'blue')
ax4.set_ylabel(r'M_ctrl')
ax4.set_xlabel(r't, [s]')
ax4.grid(True)
ax4.legend()

# ax4.set_title("B_o - B_b")
# ax4.plot(t, B_o[:, 0] - B_b[:, 0], label = '$Bo_x - Bb_x$', color = 'red')
# ax4.plot(t, B_o[:, 1] - B_b[:, 1], label = '$Bo_y - Bb_y$', color = 'green')
# ax4.plot(t, B_o[:, 2] - B_b[:, 2], label = '$Bo_z - Bb_z$', color = 'blue')
# ax4.set_ylabel(r'B_o - B_b')
# ax4.set_xlabel(r't, [s]')
# ax4.grid(True)
# ax4.legend()

# ax4.set_title("lambda_l")
# ax4.plot(t, lambda_l, label = '$\lambda_l$', color = 'red')
# ax4.set_ylabel(r'lambda_l')
# ax4.set_xlabel(r't, [s]')
# ax4.grid(True)
# ax4.legend()

ax5 = fig1.add_subplot(3,2,5)

ax5.set_title("Angular Velocity")
ax5.plot(t, x[:,4], label = '$\omega_x$', color = 'red')
ax5.plot(t, x[:,5], label = '$\omega_y$', color = 'green')
ax5.plot(t, x[:,6], label = '$\omega_z$', color = 'blue')
ax5.set_ylabel(r'angular velocity, [rad/s]')
ax5.set_xlabel(r't, [s]')
ax5.grid(True)
ax5.legend()

ax6 = fig1.add_subplot(3,2,6)

ax6.set_title("RW control torque")
ax6.plot(t, -u[:,0], label = '$h_x$', color = 'red')
ax6.plot(t, -u[:,1], label = '$h_y$', color = 'green')
ax6.plot(t, -u[:,2], label = '$h_z$', color = 'blue')
ax6.set_ylabel(r'h, [Nms]')
ax6.set_xlabel(r't, [s]')
ax6.grid(True)
ax6.legend()

plt.tight_layout()
plt.show()