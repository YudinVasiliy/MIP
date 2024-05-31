import pybullet as p
import numpy as np

dt = 1/240 # pybullet simulation step
th0 = np.deg2rad(15) #initial positin
thd = np.deg2rad(30) #destiny
jIdx = 1 #joint index
maxTime = 5 #process time
logTime = np.arange(0.0, maxTime, dt) #array with time moments
sz = len(logTime)
logPos = np.zeros(sz) #array of traj position
logPos[0] = th0
logVel = np.zeros(sz) #velocities
logAcc = np.zeros(sz) #accelerations
logCtrl = np.zeros(sz) #controls
idx = 0 #index for writing in arrays
tau = 0 #control
T = 2 #time to get to destiny
#factors
kp = 0.1
kd = 100
ki = 1
#integral discrepancy
e_int = 0

L = 0.5
g = 10
m = 1
k = 0.5

physicsClient = p.connect(p.DIRECT) # or p.DIRECT for non-graphical version
p.setGravity(0,0,-10)
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

# turn off internal damping
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetPosition=th0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

#from polynomial
a3 = 10/T**3
a4 = -15/T**4
a5 = 6/T**5

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
for t in logTime[1:]:
    jointState = p.getJointState(boxId, jIdx)
    #current position
    th1 = jointState[0]
    #current velocity
    dth1 = jointState[1]

    #current s, part of the path, [0,1]
    s = a3*t**3 + a4*t**4 + a5*t**5

    #current position, velocity and acceleration from polynomial
    theta = th0 + s*(thd - th0)
    theta_speed = ((30*t**2)/T**3 - (60*t**3)/T**4 + (30*t**4)/T**5)*(thd - th0)
    theta_acceler = (60*t/T**3 - 180*t**2/T**4 + 120*t**3/T**5)*(thd - th0)
    #discrepancy
    e = theta-th1
    #integral discrepancy
    e_int += e * dt

    u =  -kp*e - kd*(dth1 - theta_speed) - ki*e_int - theta_acceler
    if t > T:
      u = 0
      theta_acceler = 0
    #control
    tau = m*g*L*np.sin(th1) + k*dth1 + m*L*L*(u)

    logAcc[idx] = np.rad2deg(theta_acceler)
    logCtrl[idx]=tau

    p.setJointMotorControl2(
        bodyIndex=boxId,
        jointIndex=jIdx,
        controlMode=p.TORQUE_CONTROL,
        force=tau
    )

    p.stepSimulation()

    jointState = p.getJointState(boxId, jIdx)
    th1 = jointState[0]
    dth1 = jointState[1]
    logVel[idx] = np.rad2deg(dth1)
    idx += 1
    logPos[idx] = np.rad2deg(th1)

logVel[idx] = np.rad2deg(p.getJointState(boxId, jIdx)[1])
logAcc[idx] = np.rad2deg(theta_acceler)

import matplotlib.pyplot as plt

plt.subplot(4,1,1)
plt.grid(True)
plt.plot(logTime[1:], logPos[1:], label = "simPos")
plt.plot([logTime[0],logTime[-1]],[np.rad2deg(thd),np.rad2deg(thd)],'r', label='refPos')
plt.ylim((0, 35))
plt.legend()

plt.subplot(4,1,2)
plt.grid(True)
plt.plot(logTime, logVel, label = "simVel")
plt.legend()

plt.subplot(4,1,3)
plt.grid(True)
plt.plot(logTime, logAcc, label = "simAcc")
plt.legend()

plt.subplot(4,1,4)
plt.grid(True)
plt.plot(logTime, logCtrl, label = "simCtrl")
plt.legend()

plt.show()

p.disconnect()