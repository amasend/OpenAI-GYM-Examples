import gym
import time
import numpy as np
import argparse
import tensorflow as tf
import shutil
from matplotlib import pyplot as plt
from collections import deque
import os
import pandas as pd


_CONTRAST_ = np.array([210, 164, 74]).mean()
_LIFES_ = [True, True]


class MsPacman:
    """Class for building MsPacman Deep Q-Network Reinforcement Learning agent."""
    def __init__(self, n_outputs, replay_memory_size=500000, input_height=88, input_width=80, input_channels=1,
                 conv_n_maps=[32, 64, 64],
                 conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)], conv_strides=[4, 2, 1],
                 conv_paddings=["SAME"] * 3, conv_activation=[tf.nn.relu] * 3, n_hidden_in=64 * 11 * 10,
                 n_hidden=512, hidden_activation=tf.nn.relu,
                 initializer=tf.contrib.layers.variance_scaling_initializer(),
                 eps_min=0.1, eps_max=1.0, eps_decay_steps=2000000, learning_rate=0.001, momentum=0.95,
                 n_steps=4000000, training_start=100000, training_interval=16, save_steps=1000,
                 copy_steps=500, gamma = 0.99, skip_start=90, batch_size=64,
                 checkpoint_path="./my_dqn.ckpt"):

        # Reduced image size for DQN processing
        self.input_height = input_height  # How many pixels in height
        self.input_width = input_width  # How many pixels in width
        self.input_channels = input_channels  # How many color channels (if 1, set to gray scale)

        # NN hyperparameters
        self.conv_n_maps = conv_n_maps
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.conv_paddings = conv_paddings
        self.conv_activation = conv_activation
        self.n_hidden_in = n_hidden_in
        self.n_hidden = n_hidden
        self.hidden_activation = hidden_activation
        self.n_outputs = n_outputs
        self.initializer = initializer

        self.online_q_values = None  # Placeholder for online q_values
        self.online_vars = None  # Placeholder for online variables
        self.target_q_values = None  # Placeholder for target q_values
        self.target_vars = None  # Placeholder for target variables
        # Placeholders for copy operation (from online to target DQN)
        self.copy_ops = None
        self.copy_online_to_target = None

        self.replay_memory = deque([], maxlen=replay_memory_size)  # Placeholder for replay memory

        # Epsilon greedy policy parameters
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_decay_steps = eps_decay_steps

        # Hyperparameters for Nesterov Accelerated Gradient optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.n_steps = n_steps  # total number of training steps
        self.training_start = training_start  # start training after 10,000 game iterations
        self.training_interval = training_interval  # run a training step every 4 game iterations
        self.save_steps = save_steps  # save the model every 1,000 training steps
        self.copy_steps = copy_steps  # copy online DQN to target DQN every 10,000 training steps
        self.gamma = gamma  # the discount factor
        self.skip_start = skip_start  # Skip the start of every game (it's just waiting time).
        self.batch_size = batch_size
        self.iteration = 0  # game iterations
        self.checkpoint_path = checkpoint_path
        self.done = True  # env needs to be reset

        self.create_dqns()  # Initialize two DQNs for further computing
        self.compute_online_q_value()  # Initialize all online q_values
        self.compute_loss()  # Initialize loss function
        self.initialize()  # Initialize completed tensorflow graphs

    def q_network(self, x_state, name):
        """Method for building and initializing DQNs. Online and target DQNs are identical in architecture.
        The only one difference is that online DQN computes its q_values on every game step, but target DQN
        is used to control MsPacman on real time. So after a few iterations of online DQN, all vaiables
        are copied strictly into target DQN to improve steering."""
        prev_layer = x_state  # Take current state (observation)
        with tf.variable_scope(name) as scope:  # For individual network name
            # Build particular Convolution Neural Network layers
            for n_maps, kernel_size, strides, padding, activation in zip(
                    self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides,
                    self.conv_paddings, self.conv_activation):
                prev_layer = tf.layers.conv2d(
                    prev_layer, filters=n_maps, kernel_size=kernel_size,
                    strides=strides, padding=padding, activation=activation,
                    kernel_initializer=self.initializer)
            last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, self.n_hidden_in])
            hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden,
                                     activation=self.hidden_activation,
                                     kernel_initializer=self.initializer)
            outputs = tf.layers.dense(hidden, self.n_outputs,
                                      kernel_initializer=self.initializer)
        # Get all trainable variables form particular network
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=scope.name)
        # Also get theirs names
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name

    def create_dqns(self):
        """Creates DQNs and connects networks via copy online to target procedure."""
        # Placeholder for input image
        self.x_state = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width,
                                                    self.input_channels])
        # Creates two DQNs
        self.online_q_values, self.online_vars = self.q_network(self.x_state, name="q_networks/online")
        self.target_q_values, self.target_vars = self.q_network(self.x_state, name="q_networks/target")

        # Initialize copy procedure (from online to target)
        self.copy_ops = [target_var.assign(self.online_vars[var_name])
                         for var_name, target_var in self.target_vars.items()]
        self.copy_online_to_target = tf.group(*self.copy_ops)

    def compute_online_q_value(self):
        """Look at the q_value corresponded only to played action."""
        self.x_action = tf.placeholder(tf.int32, shape=[None])
        self.q_value = tf.reduce_sum(self.target_q_values * tf.one_hot(self.x_action, self.n_outputs),
                                     axis=1, keepdims=True)

    def initialize(self):
        """Build initializer for DQNs."""
        # Responsible for counting each step after warmup
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        # Build training initializer along with saver instance
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum, use_nesterov=True)
        self.training_op = optimizer.minimize(self.loss, global_step=self.global_step)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def compute_loss(self):
        """Build loss function. Bigger punishment for bigger errors."""
        self.y = tf.placeholder(tf.float32, shape=[None, 1])
        error = tf.abs(self.y - self.q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    def sample_memories(self, batch_size):
        """Sample random batch of replay memory."""
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
                cols[4].reshape(-1, 1))

    def epsilon_greedy(self, q_values, step):
        """Epsilon greedy technique for random actions implementation."""
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)  # random action
        else:
            return np.argmax(q_values)  # optimal action

    def explore(self, env):
        """The core of the agent. Initialize tf session and start exploring the game.
        Agent takes an action only every 4 frames. It waits with DQN training for warmup period and trains Online DQN
        every train period. All Online DQN variavbles are copied to target DQN every copy period.
        Each game cumulative reward is stored for further investigation."""
        with tf.Session() as sess:
            # TODO: implement replay memory restoration!
            # Restore DQN variables if checkpoints are available (but it not restore replay memory!!!)
            if os.path.isfile(self.checkpoint_path + ".index"):
                self.saver.restore(sess, self.checkpoint_path)
            else:  # Initialize both online and target DQNs
                self.init.run()
                self.copy_online_to_target.run()
            # all_rewards = []
            game_reward = 0
            global _LIFES_
            while True:
                step = self.global_step.eval()
                if step >= self.n_steps:
                    break
                self.iteration += 1
                if self.done:  # game over, start again
                    # Store game rewards into a file
                    print("Game reward: {}".format(game_reward))
                    with open('games.csv', 'a')as f:
                        f.write('{},\n'.format(game_reward))
                    game_reward = 0
                    _LIFES_ = [True, True]
                    obs = env.reset()
                    print('Step: {}'.format(step))
                    print('MsPacman is dead, environment reset!')
                    for skip in range(self.skip_start):  # skip the start of each game
                        obs, reward, done, info = env.step(0)
                    state = self.preprocess_observation(obs)
                    self.prev_action = 0
                    self.cumulative_reward = 0

                if self.iteration % 4 == 0:  # Make a decision only every 4 frames of game
                    # Online DQN evaluates what to do
                    q_values = self.online_q_values.eval(feed_dict={self.x_state: [state]})
                    action = self.epsilon_greedy(q_values, step)
                    self.prev_action = action
                    # Online DQN plays
                    obs, reward, self.done, info = env.step(action)
                    env.render()
                    next_state = self.preprocess_observation(obs)
                    reward += self.punishment_reward(next_state)  # Give minus rewards if MsPacman dies
                    self.cumulative_reward += reward
                    game_reward += self.cumulative_reward

                    # Let's memorize what just happened
                    self.replay_memory.append((state, action, self.cumulative_reward, next_state, 1.0 - self.done))
                    self.cumulative_reward = 0
                    state = next_state
                    state_prev = state

                    if self.iteration < self.training_start or self.iteration % self.training_interval != 0:
                        continue  # only train after warmup period and at regular intervals

                    # Sample memories and use the target DQN to produce the target Q-Value
                    X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                        self.sample_memories(self.batch_size))
                    next_q_values = self.target_q_values.eval(
                        feed_dict={self.x_state: X_next_state_val})
                    max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                    y_val = rewards + continues * self.gamma * max_next_q_values

                    # Train the online DQN
                    self.training_op.run(feed_dict={self.x_state: X_state_val,
                                                    self.x_action: X_action_val, self.y: y_val})
                    # Regularly copy the online DQN to the target DQN
                    if step % self.copy_steps == 0:
                        self.copy_online_to_target.run()
                        print('Online to target copied!')

                    # And save regularly
                    if step % self.save_steps == 0:
                        self.saver.save(sess, self.checkpoint_path)
                        print('Model saved!')

                else:  # Play previous action in the meantime when this is not an actual action frame
                    obs, reward, self.done, info = env.step(self.prev_action)
                    if self.done:
                        self.cumulative_reward += reward
                        game_reward += self.cumulative_reward
                        next_state = self.preprocess_observation(obs)
                        self.replay_memory.append((state_prev, self.prev_action,
                                                   self.cumulative_reward, next_state, 1.0 - self.done))
                    else:
                        self.cumulative_reward += reward
                        state = self.preprocess_observation(obs)

    @staticmethod
    def preprocess_observation(obs):
        """Scale and preprocess observation to gray scale."""
        img = obs[1:176:2, ::2]  # crop and downsize
        img = img.mean(axis=2)  # to grayscale
        img[img == _CONTRAST_] = 0  # improve contrast
        img = (img - 128) / 128 - 1  # normalize from -1. to 1.
        return img.reshape(88, 80, 1)

    @staticmethod
    def punishment_reward(img):
        """Implements punishment if MsPacman dies."""
        global _LIFES_
        life_pixels = (img.reshape(88, 80)[-1, 8], img.reshape(88, 80)[-1, 16])
        if (life_pixels[0] != -2.0) and (life_pixels[1] != -2.0):
            return 0
        elif (life_pixels[0] != -2.0) and _LIFES_[1]:
            _LIFES_[1] = False
            return -20
        elif (life_pixels[0] != -2.0) and not _LIFES_[1]:
            return 0
        elif _LIFES_[0]:
            _LIFES_[0] = False
            return -20
        else:
            return 0


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    pacman = MsPacman(n_outputs=9)
    pacman.explore(env=env)
