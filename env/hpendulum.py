from .pendulum import Pendulum
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
    

class HPendulum:
    ''' Hybrid Pendulum environment. Joint angle, velocity are continuous,
        torque are discretized with the specified steps. Torque is saturated.
        Guassian noise can be added in the dynamics.
    '''
    def __init__(self, nu=11, uMax=5, dt=0.2, ndt=1, noise_stddev=0):
        self.pendulum = Pendulum(1,noise_stddev)
        self.pendulum.DT  = dt
        self.pendulum.NDT = ndt
        self.nu = nu        # Number of discretization steps for joint torque
        self.nx = self.pendulum.nx
        self.uMax = uMax    # Max torque (u in [-umax,umax])
        self.DU = 2*uMax/nu # discretization resolution for joint torque
    
    # Continuous to discrete
    def c2du(self, u):
        u = np.clip(u,-self.uMax+1e-3,self.uMax-1e-3)
        return int(np.floor((u+self.uMax)/self.DU))
    
    # Discrete to continuous
    def d2cu(self, iu):
        iu = np.clip(iu,0,self.nu-1) - (self.nu-1)/2
        return iu*self.DU
    
    def reset(self):
        self.x = self.pendulum.reset()
        return self.x

    def step(self,iu):
        u = self.d2cu(iu)
        self.x, cost = self.pendulum.step(u)
        return self.x, cost
    
    def render(self):
        self.pendulum.render()

if __name__=="__main__":
    print("Start tests")
    env = HPendulum()
    nu = env.nu
    
    x = env.reset()
    print('initial state x:\n', x)
    for iu in range(nu):
        x_next, cost = env.step(iu)
        env.render()
        print('next state x:\n', x_next)
