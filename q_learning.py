'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
'''

import numpy as np
from numpy.random import randint, uniform
from env.dpendulum import DPendulum
import matplotlib.pyplot as plt
import time

def q_learning(env, gamma, Q, nEpisodes, maxEpisodeLength, 
               learningRate, exploration_prob, exploration_decreasing_decay,
               min_exploration_prob, compute_V_pi_from_Q, plot=False, nprint=1000):
    ''' TD(0) Policy Evaluation:
        env: environment 
        gamma: discount factor
        Q: initial guess for Q table
        nEpisodes: number of episodes to be used for evaluation
        maxEpisodeLength: maximum length of an episode
        learningRate: learning rate of the algorithm
        exploration_prob: initial exploration probability for epsilon-greedy policy
        exploration_decreasing_decay: rate of exponential decay of exploration prob
        min_exploration_prob: lower bound of exploration probability
        compute_V_pi_from_Q: function to compute V and pi from Q
        plot: if True plot the V table every nprint iterations
        nprint: print some info every nprint iterations
    '''
    h_ctg = [] # Learning history (for plot).
    Q_old = np.copy(Q)
    for episode in range(1,nEpisodes):
        x    = env.reset()
        costToGo = 0.0
        gamma_i = 1
        for steps in range(maxEpisodeLength):
            if uniform(0,1) < exploration_prob:
                u = randint(env.nu)
            else:
                u = np.argmin(Q[x,:]) # Greedy action
                # u = np.argmin(Q[x,:] + np.random.randn(1,NU)/episode) # Greedy action with noise
            x_next,cost = env.step(u)
    
            # Compute reference Q-value at state x respecting HJB
            Qref = cost + gamma*np.min(Q[x_next,:])
    
            # Update Q-Table to better fit HJB
            Q[x,u]      += learningRate*(Qref-Q[x,u])
            x           = x_next
            costToGo    += gamma_i*cost
            gamma_i     *= gamma
    
        exploration_prob = max(min_exploration_prob, 
                               np.exp(-exploration_decreasing_decay*episode))
        h_ctg.append(costToGo)
        if not episode%nprint: 
            print('Episode #%d done with cost %d and %.1f exploration prob' % (
                  episode, np.mean(h_ctg[-nprint:]), 100*exploration_prob))
            print("max|Q - Q_old|=%.2f"%(np.max(np.abs(Q-Q_old))))
            print("avg|Q - Q_old|=%.2f"%(np.mean(np.abs(Q-Q_old))))
            if(plot):
                # env.plot_Q_table(Q)
                # render_greedy_policy(env, Q)
                V, pi = compute_V_pi_from_Q(env, Q)
                env.plot_V_table(V)
                # env.plot_policy(pi)
            Q_old = np.copy(Q)
    
    return Q, h_ctg

def render_greedy_policy(env, Q, gamma, x0=None, maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    x0 = x = env.reset(x0)
    costToGo = 0.0
    gamma_i = 1
    for i in range(maxiter):
        u = np.argmin(Q[x,:])
        # print("State", x, "Control", u, "Q", Q[x,u])
        x,c = env.step(u)
        costToGo += gamma_i*c
        gamma_i *= gamma
        env.render()
    print("Real cost to go of state", x0, ":", costToGo)
    
def compute_V_pi_from_Q(env, Q):
    ''' Compute Value table and greedy policy pi from Q table. '''
    V = np.zeros(Q.shape[0])
    pi = np.zeros(Q.shape[0], np.int)
    for x in range(Q.shape[0]):
        # pi[x] = np.argmin(Q[x,:])
        # Rather than simply using argmin we do something slightly more complex
        # to ensure simmetry of the policy when multiply control inputs
        # result in the same value. In these cases we prefer the more extreme
        # actions
        V[x] = np.min(Q[x,:])
        u_best = np.where(Q[x,:]==V[x])[0]
        if(u_best[0]>env.c2du(0.0)):
            pi[x] = u_best[-1]
        elif(u_best[-1]<env.c2du(0.0)):
            pi[x] = u_best[0]
        else:
            pi[x] = u_best[int(u_best.shape[0]/2)]

    return V, pi


if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    NEPISODES               = 5000          # Number of training episodes
    NPRINT                  = 500           # print something every NPRINT episodes
    MAX_EPISODE_LENGTH      = 100           # Max episode length
    LEARNING_RATE           = 0.8           # alpha coefficient of Q learning algorithm
    DISCOUNT                = 0.9           # Discount factor 
    PLOT                    = True          # Plot stuff if True
    exploration_prob                = 1     # initial exploration probability of eps-greedy policy
    exploration_decreasing_decay    = 0.001 # exploration decay for exponential decreasing
    min_exploration_prob            = 0.001 # minimum of exploration proba
    
    ### --- Environment
    nq=51   # number of discretization steps for the joint angle q
    nv=21   # number of discretization steps for the joint velocity v
    nu=11   # number of discretization steps for the joint torque u
    env = DPendulum(nq, nv, nu)
    Q   = np.zeros([env.nx,env.nu])       # Q-table initialized to 0
    
    ### --- Q-learning
    Q, h_ctg = q_learning(env, DISCOUNT, Q, NEPISODES, MAX_EPISODE_LENGTH, 
                          LEARNING_RATE, exploration_prob, exploration_decreasing_decay,
                          min_exploration_prob, compute_V_pi_from_Q, PLOT, NPRINT)
    
    ### --- print
    print("\nTraining finished")
    V, pi = compute_V_pi_from_Q(env,Q)
    env.plot_V_table(V)
    env.plot_policy(pi)
    print("Average/min/max Value:", np.mean(V), np.min(V), np.max(V)) 

    print("\nTotal rate of success: %.3f" % (-sum(h_ctg)/NEPISODES))
    render_greedy_policy(env, Q, DISCOUNT)
    plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES) )
    plt.title ("Average cost-to-go")
    plt.show()