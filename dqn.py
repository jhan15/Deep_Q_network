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
BATCH_SIZE             = 32            # Batch size sampled from replay buffer
REPLAY_BUFFER_SIZE     = 50000         # Size of replay buffer
MIN_BUFFER_SIZE        = 1000          # Minimum buffer size to start training
UPDATE_Q_TARGET_STEPS  = 100           # Steps to update Q target
NEPISODES              = 5000          # Number of training episodes
MAX_EPISODE_LENGTH     = 100           # Max episode length
QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
GAMMA                  = 0.9           # Discount factor 
EPSILON                = 1             # Initial exploration probability of eps-greedy policy
EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
MIN_EPSILON            = 0.001         # Minimum of exploration probability


class DQN:
    
    def __init__(self, nu):
        self.nu = nu
        self.env = HPendulum(nu)
        self.nx = self.env.nx

    def get_critic(self):
        ''' Create the neural network to represent the Q function '''
        inputs = layers.Input(shape=(self.nx))
        state_out1 = layers.Dense(16, activation="relu")(inputs) 
        state_out2 = layers.Dense(32, activation="relu")(state_out1) 
        state_out3 = layers.Dense(64, activation="relu")(state_out2) 
        state_out4 = layers.Dense(64, activation="relu")(state_out3)
        outputs = layers.Dense(self.nu)(state_out4)
    
        model = tf.keras.Model(inputs, outputs)
    
        return model

    def update(self, batch):
        ''' Update the weights of the Q network using the specified batch of data '''
        x_batch      = np.array([sample[0] for sample in batch]).squeeze()
        u_batch      = np.array([sample[1] for sample in batch])
        cost_batch   = np.array([sample[2] for sample in batch])
        x_next_batch = np.array([sample[3] for sample in batch]).squeeze()
        done_batch   = np.array([sample[4] for sample in batch])
        
        with tf.GradientTape() as tape:
            target_value = np.min(self.Q_target(x_next_batch, training=True), axis=1)
            
            # Compute 1-step targets for the critic loss
            y = np.zeros(len(batch))
            for id, done in enumerate(done_batch):
                if done:
                    y[id] = cost_batch[id]
                else:
                    y[id] = cost_batch[id] + GAMMA*target_value[id]            
            
            Q_output = self.Q(x_batch, training=True)
            Q_value  = Q_output[np.arange(len(batch)), u_batch]
            Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
        
        Q_grad = tape.gradient(Q_loss, self.Q.trainable_variables)
        self.optimizer.apply_gradients(zip(Q_grad, self.Q.trainable_variables))

    def pick_control(self, x, epsilon):
        ''' Pick control by epsilon-greedy strategy '''
        if uniform(0,1) < epsilon:
            u = randint(self.nu)
        else:
            u = np.argmin(self.Q.predict(x.T))
        
        return u

    def render(self, maxiter=100):
        '''Roll-out from random state using trained DQN.'''
        x0 = x = self.env.reset()
        costToGo = 0.0
        gamma_i = 1
        
        for i in range(maxiter):
            u = np.argmin(self.Q.predict(x.T))
            x, cost = self.env.step(u)
            costToGo += gamma_i*cost
            gamma_i *= GAMMA
            self.env.render()
        
        print("Real cost to go of state", x0.squeeze(), ":", costToGo)

    def learning(self, nprint=100):
        ''' Learning of DQN algorithm '''
        self.Q = self.get_critic()
        self.Q.summary()
        
        self.Q_target = self.get_critic()
        self.Q_target.set_weights(self.Q.get_weights())
        
        self.optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
        self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
        
        self.h_ctg = []
        self.best_ctg = np.inf
        
        steps = 0
        epsilon = EPSILON
        
        for episode in range(NEPISODES):
            cost_to_go = 0.0
            x = self.env.reset()
            gamma_i = 1
            
            for step in range(MAX_EPISODE_LENGTH):
                u = self.pick_control(x, epsilon)
                x_next, cost = self.env.step(u)
                
                done = True if step == MAX_EPISODE_LENGTH - 1 else False
                self.replay_buffer.append([x, u, cost, x_next, done])
                
                if steps % UPDATE_Q_TARGET_STEPS == 0:
                    # Regularly update weights of target network
                    self.Q_target.set_weights(self.Q.get_weights())
                
                if len(self.replay_buffer) > MIN_BUFFER_SIZE and steps % SAMPLING_STEPS == 0:
                    # Sampling from replay buffer and train
                    batch = sample(self.replay_buffer, BATCH_SIZE)
                    self.update(batch)
                
                x = x_next
                cost_to_go += gamma_i * cost
                gamma_i *= GAMMA
                steps += 1
            
            if cost_to_go < best_ctg:
                # Save NN weights to file (in HDF5)
                self.Q.save_weights("model/dqn.h5")
                best_ctg = cost_to_go
            
            epsilon = max(MIN_EPSILON, np.exp(-EPSILON_DECAY*episode))
            h_ctg.append(cost_to_go)
            
            if episode % nprint == 0:
                print('Episode #%d done with cost %d and %.1f exploration prob' % (
                      episode, np.mean(h_ctg[-nprint:]), 100*epsilon))
    
    def plot_h_ctg(self):
        ''' Plot the average cost-to-go history '''
        plt.plot(np.cumsum(self.h_ctg)/range(1,NEPISODES+1))
        plt.title ("Average cost-to-go")
        plt.show()

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    nu = 11
    dqn = DQN(nu=11)
    
    train = True
    if train:
        dqn.learning()
        dqn.plot_h_ctg()
    
    # Load NN weights from file
    dqn.Q.load_weights("model/dqn.h5")
    dqn.render()
    