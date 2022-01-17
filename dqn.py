import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform
from env.hpendulum import HPendulum
import time
from collections import deque
import matplotlib.pyplot as plt
from random import sample

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


### --- Hyper paramaters
SAMPLING_STEPS         = 4             # Steps to sample from replay buffer
BATCH_SIZE             = 64            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 50000         # Size of replay buffer
MIN_BUFFER_SIZE        = 5000          # Minimum buffer size to start training
UPDATE_Q_TARGET_STEPS  = 100           # Steps to update Q target
NEPISODES              = 3000          # Number of training episodes
MAX_EPISODE_LENGTH     = 200           # Max episode length
QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
MIN_EPSILON            = 0.001         # Minimum of exploration probability


class DQN:
    ''' Deep Q Network algorithm '''
    
    def __init__(self, nu, nbJoint=1, dt=0.1):
        self.nu = nu
        self.nbJoint = nbJoint
        self.env = HPendulum(self.nbJoint, self.nu, dt=dt)
        self.nx = self.env.nx
        
        self.Q = self.get_critic()
        self.Q.summary()

    def get_critic(self):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(self.nx))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.nbJoint*self.nu)(state_out4)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def update(self, batch):
        ''' Update the weights of the Q network using the specified batch of data '''
        x_batch      = np.array([sample[0] for sample in batch])
        u_batch      = np.array([sample[1] for sample in batch])
        cost_batch   = np.array([sample[2] for sample in batch])
        x_next_batch = np.array([sample[3] for sample in batch])
        done_batch   = np.array([sample[4] for sample in batch])
        
        n = len(batch)
        
        with tf.GradientTape() as tape:
            # Compute Q target
            target_output = self.Q_target(x_next_batch, training=True).reshape((n,-1,self.nbJoint))
            target_value  = tf.math.reduce_sum(np.min(target_output, axis=1), axis=1)
            
            # Compute 1-step targets for the critic loss
            y = np.zeros(n)
            for id, done in enumerate(done_batch):
                if done:
                    y[id] = cost_batch[id]
                else:
                    y[id] = cost_batch[id] + GAMMA*target_value[id]      
            
            # Compute Q
            Q_output = self.Q(x_batch, training=True).reshape((n,-1,self.nbJoint))
            d1 = np.repeat(np.arange(n),self.nbJoint).reshape(n,-1)
            d2 = u_batch.reshape(n,-1)
            d3 = np.repeat(np.arange(self.nbJoint).reshape(1,-1),n,axis=0)
            Q_value  = tf.math.reduce_sum(Q_output[d1, d2, d3], axis=1)
            
            # Compute Q loss
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
        
        Q_grad = tape.gradient(Q_loss, self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(Q_grad, self.Q.trainable_variables))

    def pick_control(self, x, epsilon):
        ''' Pick control by epsilon-greedy strategy '''
        if uniform(0,1) < epsilon:
            u = randint(self.nu, size=self.nbJoint)
        else:
            pred = self.Q.predict(x.reshape(1,-1))
            u = np.argmin(pred.reshape(self.nu,self.nbJoint), axis=0)
    
        if len(u) == 1:
            u = u[0]
        return u

    def learning(self, nprint=100):
        ''' Learning of DQN algorithm '''
        self.Q_target = self.get_critic()
        self.Q_target.set_weights(self.Q.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        
        self.h_ctg = []
        self.best_ctg = np.inf
        
        steps = 0
        epsilon = EPSILON
        
        t = time.time()
        
        for episode in range(NEPISODES):
            cost_to_go = 0.0
            x = self.env.reset()
            gamma_i = 1
            
            for step in range(MAX_EPISODE_LENGTH):
                u = self.pick_control(x, epsilon)
                x_next, cost = self.env.step(u)
                
                done = True if step == MAX_EPISODE_LENGTH - 1 else False
                self.replay_buffer.append([x, u, cost, x_next, done])
                
                # Regularly update weights of target network
                if steps % UPDATE_Q_TARGET_STEPS == 0:
                    self.Q_target.set_weights(self.Q.get_weights())
                
                # Sampling from replay buffer and train
                if len(self.replay_buffer) > MIN_BUFFER_SIZE and steps % SAMPLING_STEPS == 0:
                    batch = sample(self.replay_buffer, BATCH_SIZE)
                    self.update(batch)
                
                x = x_next
                cost_to_go += gamma_i * cost
                gamma_i *= GAMMA
                steps += 1
            
            # Save NN weights to file (in HDF5)
            if cost_to_go < self.best_ctg:
                self.Q.save_weights("model/dqn.h5")
                self.best_ctg = cost_to_go
            
            epsilon = max(MIN_EPSILON, np.exp(-EPSILON_DECAY*episode))
            self.h_ctg.append(cost_to_go)
            
            if episode % nprint == 0:
                dt = time.time() - t
                t = time.time()
                print('Episode #%d done with cost %.1f and %.1f exploration prob, used %.1f s' % (
                      episode, np.mean(self.h_ctg[-nprint:]), 100*epsilon, dt))
        
        self.plot_h_ctg()
    
    def plot_h_ctg(self):
        ''' Plot the average cost-to-go history '''
        plt.plot(np.cumsum(self.h_ctg)/range(1,NEPISODES+1))
        plt.title ("Average cost-to-go")
        plt.show()
    
    def render(self, x=None, maxiter=150):
        '''Roll-out from random state using trained DQN.'''
        # Load NN weights from file
        self.Q.load_weights("model/dqn.h5")
        
        if x is None:
            x0 = x = self.env.reset()
        else:
            x0 = x
        
        costToGo = 0.0
        gamma_i = 1
        
        for i in range(maxiter):
            pred = self.Q.predict(x.reshape(1,-1))
            u = np.argmin(pred.reshape(self.nu,self.nbJoint), axis=0)
            if len(u) == 1:
                u = u[0]
            x, cost = self.env.step(u)
            costToGo += gamma_i*cost
            gamma_i *= GAMMA
            self.env.render()
        
        print("Real cost to go of state", x0, ":", costToGo)

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    nu = 11
    nbJoint = 2
    dqn = DQN(nu,nbJoint)
    
    #dqn.learning()
    dqn.render()
