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
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()

def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+nu))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs)

    return model

def update(xu_batch, cost_batch, xu_next_batch, Q, Q_target, critic_optimizer, gamma):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager
        # and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable,
        # where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_next_batch, training=True)   
        # Compute 1-step targets for the critic loss
        y = cost_batch + gamma*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean
        # of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))

def pick_control_epsilon_greedy(nu, Q, x, u_, epsilon):
    ''' Pick control by epsilon-greedy strategy '''
    if uniform(0,1) < epsilon:
        u = randint(nu)
    else:
        u = pick_control_by_dqn(nu, Q, x, u_)
    
    return u

def pick_control_by_dqn(nu, Q, x, u_):
    ''' Pick the control by DQN '''
    x_  = np.repeat(x, nu, axis=1)
    xu_ = np.concatenate((x_,u_))
    y_  = Q.predict(xu_.T)
    u   = np.argmin(y_)
    
    return u

def u2one_hot(nu, u):
    ''' Convert u to one-hot encoding '''
    u = u if type(u) is np.ndarray else np.array([u])
    return np.eye(nu)[u].T

def preprocess_batch(batch, nu):
    ''' Preprocess the batch data for training '''
    x_batch = np.array([sample[0] for sample in batch])
    u_batch = np.array([sample[1] for sample in batch])
    cost_batch = np.array([sample[2] for sample in batch])
    x_next_batch = np.array([sample[3] for sample in batch])
    
    x_batch = x_batch.squeeze().T
    u_batch = u2one_hot(nu, u_batch)
    x_next_batch = x_next_batch.squeeze().T
    
    xu_batch = np.concatenate((x_batch,u_batch)).T
    xu_next_batch = np.concatenate((x_next_batch,u_batch)).T
    
    return xu_batch, cost_batch, xu_next_batch

def render_dqn(env, Q, gamma, maxiter=200):
    '''Roll-out from random state using trained DQN.'''
    x0 = x = env.reset()
    costToGo = 0.0
    gamma_i = 1
    u_one_hot = u2one_hot(env.nu, np.arange(env.nu))
    
    for i in range(maxiter):
        u = pick_control_by_dqn(env.nu, Q, x, u_one_hot)
        x, cost = env.step(u)
        costToGo += gamma_i*cost
        gamma_i *= gamma
        env.render()
    print("Real cost to go of state", x0.squeeze(), ":", costToGo)

def DQN_learning(env, Q, Q_target, gamma, nEpisodes, maxEpisodeLength,
                 epsilon, epsilon_decay, min_epsilon,
                 batch_size, sampling_steps, update_Q_target_steps,
                 replay_buffer, min_buffer_size, critic_optimizer, nprint=100):
    
    h_ctg = []
    best_ctg = np.inf
    steps = 0
    u_one_hot = u2one_hot(env.nu, np.arange(env.nu))
    
    for episode in range(nEpisodes):
        cost_to_go = 0.0
        x = env.reset()
        gamma_i = 1
        
        for step in range(maxEpisodeLength):
            steps += 1
            u = pick_control(env.nu, Q, x, u_one_hot, epsilon)
            x_next, cost = env.step(u)
            
            replay_buffer.append([x, u, cost, x_next])
            
            if len(replay_buffer) > min_buffer_size and steps % sampling_steps == 0:
                # Sampling from replay buffer and train
                batch = sample(replay_buffer, batch_size)
                xu_batch, cost_batch, xu_next_batch = preprocess_batch(batch, env.nu)
                update(xu_batch, cost_batch, xu_next_batch, Q, Q_target, critic_optimizer, gamma)
                
            if steps % update_Q_target_steps == 0:
                # Regularly update weights of target network
                Q_target.set_weights(Q.get_weights())
            
            x = x_next
            cost_to_go += gamma_i * cost
            gamma_i *= gamma
        
        if cost_to_go < best_ctg:
            # Save NN weights to file (in HDF5)
            Q.save_weights("model/dqn.h5")
            best_ctg = cost_to_go
        
        epsilon = max(min_epsilon, np.exp(-epsilon_decay*episode))
        h_ctg.append(cost_to_go)
        
        if episode % nprint == 0:
            print('Episode #%d done with cost %d and %.1f exploration prob' % (
                  episode, np.mean(h_ctg[-nprint:]), 100*epsilon))
    
    return Q, h_ctg

if __name__=='__main__':
    ### --- Random seed
    RANDOM_SEED = int((time.time()%10)*1000)
    print("Seed = %d" % RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    ### --- Hyper paramaters
    SAMPLING_STEPS         = 4             # Steps to sample from replay buffer
    BATCH_SIZE             = 32            # Batch size sampled from replay buffer
    REPLAY_BUFFER_SIZE     = 50000         # Size of replay buffer
    MIN_BUFFER_SIZE        = 1000          # Minimum buffer size to start training
    UPDATE_Q_TARGET_STEPS  = 100           # Steps to update Q target
    NEPISODES              = 5000           # Number of training episodes
    MAX_EPISODE_LENGTH     = 100           # Max episode length
    QVALUE_LEARNING_RATE   = 0.001         # Learning rate of DQN
    GAMMA                  = 0.9           # Discount factor 
    EPSILON                = 1             # Initial exploration probability of eps-greedy policy
    EPSILON_DECAY          = 0.001         # Exploration decay for exponential decreasing
    MIN_EPSILON            = 0.001         # Minimum of exploration probability
    
    ### --- Environment
    nu=11   # number of discretization steps for the joint torque u
    env = HPendulum(nu)
    nx = env.nx
    
    # Create critic and target NNs
    Q = get_critic(nx, nu)
    Q.summary()
    Q_target = get_critic(nx, nu)
    
    # Set initial weights of targets equal to those of actor and critic
    Q_target.set_weights(Q.get_weights())
    
    # Set optimizer specifying the learning rates
    critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)
    
    # Set replay buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    
#    Q, h_ctg = DQN_learning(env, Q, Q_target, GAMMA, NEPISODES, MAX_EPISODE_LENGTH,
#                            EPSILON, EPSILON_DECAY, MIN_EPSILON,
#                            BATCH_SIZE, SAMPLING_STEPS, UPDATE_Q_TARGET_STEPS,
#                            replay_buffer, MIN_BUFFER_SIZE, critic_optimizer)
    
#    plt.plot( np.cumsum(h_ctg)/range(1,NEPISODES+1) )
#    plt.title ("Average cost-to-go")
#    plt.show()
    
    # Load NN weights from file
    Q.load_weights("model/dqn.h5")
    render_dqn(env, Q, GAMMA)
