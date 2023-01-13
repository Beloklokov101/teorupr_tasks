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
    q0, q1, q2, q3 = q0/norm_q, q1/norm_q, q2/norm_q, q3/norm_q 
    

    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1**2 + q2**2))*180/np.pi
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))*180/np.pi
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2**2 + q3**2))*180/np.pi

    return [roll, pitch, yaw]

def quat_conjugate(q):

    q_new = np.copy(q)
    q_new[1:] *= -1.

    return q_new


def ctrl_torque(quat, omega, h, ctrl, J):
       
    dw = omega - ctrl.omega_req
    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    trq_ctrl = np.zeros(3)
    
    trq_ctrl = cross_product(omega, J.dot(omega)  + h) - ctrl.kw * J.dot(dw) - ctrl.kq * J.dot(dq[1:])
            
    if np.max(np.abs(trq_ctrl)) > ctrl.trq_max:
        trq_ctrl *= ctrl.trq_max / np.max(np.abs(trq_ctrl))  
        
    for i in range(3):
        if np.abs(h[i]) / ctrl.h_max > 0.99 and trq_ctrl[i] * h[i] < 0:
            trq_ctrl[i] = 0        
        
    return trq_ctrl

def rhs(t, x, sat, ctrl):

    quat = x[:4] / np.linalg.norm(x[:4])
    omega = x[4:7]
    h = x[7:]
     
    control_action = ctrl_torque(quat, omega, h, ctrl, sat.J)        

    x_dot = np.zeros(10)

    x_dot[0] = -0.5 * quat[1:].dot(omega)
    x_dot[1:4] = 0.5 * (quat[0] * omega + cross_product(quat[1:], omega))
    x_dot[4:7] = sat.J_inv.dot(control_action - cross_product(omega, sat.J.dot(omega) + h))
    x_dot[7:]= -control_action
        
    return x_dot


def ctrl_torque_a(quat, omega, h, ctrl, J):
       
    dw = omega - ctrl.omega_req
    dq = quat_product(quat_conjugate(ctrl.q_req), quat)
    trq_ctrl = np.zeros(3)
    
    trq_gyro = cross_product(omega, J.dot(omega) + h)
    
    e_f = rotate_vec_with_quat(quat, ctrl.e_f)
    
    barrier_w = np.cos(ctrl.alpha_f) -  ctrl.e_o.dot(e_f)
    
    barrier_0 = cross_product(ctrl.e_o, e_f) * (dw.dot(J.dot(dw)) + 2 * ctrl.kq * (1 - dq[0])) / barrier_w
    
    trq_ctrl =  trq_gyro - ctrl.kq * J.dot(dq[1:]) - ctrl.kw * J.dot(dw) * barrier_w / ctrl.lambda_f  - barrier_0
            
    if np.max(np.abs(trq_ctrl)) > ctrl.trq_max:
        trq_ctrl *= ctrl.trq_max / np.max(np.abs(trq_ctrl))  
        
    for i in range(3):
        if np.abs(h[i]) / ctrl.h_max > 0.99 and trq_ctrl[i] * h[i] < 0:
            trq_ctrl[i] = 0        
        
    return trq_ctrl

def rhs_a(t, x, sat, ctrl):

    quat = x[:4] / np.linalg.norm(x[:4])
    omega = x[4:7]
    h = x[7:]
     
    control_action = ctrl_torque_a(quat, omega, h, ctrl, sat.J)        

    x_dot = np.zeros(10)

    x_dot[0] = -0.5 * quat[1:].dot(omega)
    x_dot[1:4] = 0.5 * (quat[0] * omega + cross_product(quat[1:], omega))
    x_dot[4:7] = sat.J_inv.dot(control_action - cross_product(omega, sat.J.dot(omega) + h))
    x_dot[7:]= -control_action
        
    return x_dot


class Parameters(object):
    pass

sat = Parameters()
sat.J = np.diag(np.array([3, 4, 5]))
sat.J_inv = np.linalg.inv(sat.J)

a = 0.32
ctrl = Parameters()
ctrl.kw = 2 * a
ctrl.kq = 2 * a**2
ctrl.trq_max = 0.05
ctrl.h_max = 0.5
# ctrl.omega_req = np.zeros(3)
ctrl.omega_req = np.array([0., 0.1, 0.])
ctrl.q_req = np.array([1., 0., 0., 0.])
# ctrl.q_req = np.array([np.cos(np.pi/3), 0.1, np.sin(np.pi/6), 0])

alpha_0 = np.pi / 4
x_0 = np.array([np.cos(alpha_0), 0, np.sin(alpha_0), 0, 0, 0, 0, 0, 0, 0])
# x_0 = np.array([1., 0., 0., 0, 0, 0, 0, 0, 0, 0])

t0 = 0.
tf = 50.
t_eval = np.arange(t0, tf, 0.01)

sol = solve_ivp(lambda t, x: rhs(t, x, sat, ctrl), (t0,tf), x_0, t_eval = t_eval)
x = sol.y.T
t = sol.t
roll, pitch, yaw = quat2rpy_deg(x[1:,0], x[1:,1], x[1:,2], x[1:,3])

control = np.zeros((np.size(t), 3))
for i in range(np.size(t)):
    control[i] = ctrl_torque(x[i, 0:4], x[i, 4:7], x[i, 7:], ctrl, sat.J)


roll, pitch, yaw = quat2rpy_deg(x[:,0], x[:,1], x[:,2], x[:,3])
   
fig1 = plt.figure(figsize=(16,8))
ax1 = fig1.add_subplot(2,2,1)

ax1.set_title("Euler Angles")
ax1.plot(t, roll, label = 'roll', color = 'red')
ax1.plot(t, pitch, label = 'pitch', color = 'green')
ax1.plot(t, yaw, label = 'yaw', color = 'blue')
ax1.set_ylabel(r'angles, [deg]')
ax1.set_xlabel(r't, [s]')
ax1.grid(True)
ax1.legend()

ax2 = fig1.add_subplot(2,2,2)

ax2.set_title("Angular Velocity")
ax2.plot(t, x[:,4], label = '$\omega_x$', color = 'red')
ax2.plot(t, x[:,5], label = '$\omega_y$', color = 'green')
ax2.plot(t, x[:,6], label = '$\omega_z$', color = 'blue')
ax2.set_ylabel(r'angular velocity, [rad/s]')
ax2.set_xlabel(r't, [s]')
ax2.grid(True)
ax2.legend()

ax3 = fig1.add_subplot(2,2,3)

ax3.set_title("RW angular momentum")
ax3.plot(t, x[:,7], label = '$h_x$', color = 'red')
ax3.plot(t, x[:,8], label = '$h_y$', color = 'green')
ax3.plot(t, x[:,9], label = '$h_z$', color = 'blue')
ax3.set_ylabel(r'h, [Nms]')
ax3.set_xlabel(r't, [s]')
ax3.grid(True)
ax3.legend()

ax4 = fig1.add_subplot(2,2,4)

ax4.set_title("RW control torque")
ax4.plot(t, -control[:,0], label = '$h_x$', color = 'red')
ax4.plot(t, -control[:,1], label = '$h_y$', color = 'green')
ax4.plot(t, -control[:,2], label = '$h_z$', color = 'blue')
ax4.set_ylabel(r'h, [Nms]')
ax4.set_xlabel(r't, [s]')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.legend()
plt.show()


# class Parameters(object):
#     pass

# sat = Parameters()
# sat.J = np.diag(np.array([3, 4, 5]))
# sat.J_inv = np.linalg.inv(sat.J)

# a = 0.32
# ctrl_a = Parameters()
# ctrl_a.kw = 2 * a
# ctrl_a.kq = 2 * a**2
# ctrl_a.trq_max = 0.05
# ctrl_a.h_max = 0.5
# ctrl_a.omega_req = np.zeros(3)
# ctrl_a.q_req = np.array([1., 0., 0., 0.])
# ctrl_a.e_o = np.array([1., 0., 0.])
# ctrl_a.e_f = normalize(np.array([1., 0., 1.]))
# ctrl_a.alpha_f = 0.2
# ctrl_a.lambda_f = 1

# alpha_0 = np.pi / 4
# x_0 = np.array([np.cos(alpha_0), 0, np.sin(alpha_0), 0, 0, 0, 0, 0, 0, 0])

# t0 = 0.
# tf = 80.
# t_eval = np.arange(t0, tf, 0.01)

# sol_a = solve_ivp(lambda t, x: rhs_a(t, x, sat, ctrl_a), (t0, tf), x_0, t_eval = t_eval)
# x_a = sol_a.y.T
# t_a = sol_a.t

# control_a = np.zeros((np.size(t_a), 3))
# for i in range(np.size(t_a)):
#     control_a[i] = ctrl_torque_a(x_a[i, 0:4], x_a[i, 4:7], x_a[i, 7:], ctrl_a, sat.J)


# roll, pitch, yaw = quat2rpy_deg(x_a[:,0], x_a[:,1], x_a[:,2], x_a[:,3])
   
# fig1 = plt.figure(figsize=(16,8))
# ax1 = fig1.add_subplot(2,2,1)

# ax1.set_title("Euler Angles")
# ax1.plot(t_a, roll, label = 'roll', color = 'red')
# ax1.plot(t_a, pitch, label = 'pitch', color = 'green')
# ax1.plot(t_a, yaw, label = 'yaw', color = 'blue')
# ax1.set_ylabel(r'angles, [deg]')
# ax1.set_xlabel(r't, [s]')
# ax1.grid(True)
# ax1.legend()

# ax2 = fig1.add_subplot(2,2,2)

# ax2.set_title("Angular Velocity")
# ax2.plot(t_a, x_a[:,4], label = '$\omega_x$', color = 'red')
# ax2.plot(t_a, x_a[:,5], label = '$\omega_y$', color = 'green')
# ax2.plot(t_a, x_a[:,6], label = '$\omega_z$', color = 'blue')
# ax2.set_ylabel(r'angular velocity, [rad/s]')
# ax2.set_xlabel(r't, [s]')
# ax2.grid(True)
# ax2.legend()

# ax3 = fig1.add_subplot(2,2,3)

# ax3.set_title("RW angular momentum")
# ax3.plot(t_a, x_a[:,7], label = '$h_x$', color = 'red')
# ax3.plot(t_a, x_a[:,8], label = '$h_y$', color = 'green')
# ax3.plot(t_a, x_a[:,9], label = '$h_z$', color = 'blue')
# ax3.set_ylabel(r'h, [Nms]')
# ax3.set_xlabel(r't, [s]')
# ax3.grid(True)
# ax3.legend()

# ax4 = fig1.add_subplot(2,2,4)

# ax4.set_title("RW control torque")
# ax4.plot(t_a, -control_a[:,0], label = '$h_x$', color = 'red')
# ax4.plot(t_a, -control_a[:,1], label = '$h_y$', color = 'green')
# ax4.plot(t_a, -control_a[:,2], label = '$h_z$', color = 'blue')
# ax4.set_ylabel(r'h, [Nms]')
# ax4.set_xlabel(r't, [s]')
# ax4.grid(True)
# ax4.legend()

# plt.tight_layout()

