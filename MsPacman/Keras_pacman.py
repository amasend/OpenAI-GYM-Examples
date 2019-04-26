import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from copy import deepcopy
from keras.models import load_model


_CONTRAST_ = np.array([210, 164, 74]).mean()


class MsPacman:
    """Class for building MsPacman Deep Q-Network Reinforcement Learning agent."""
    def __init__(self, n_outputs, replay_memory_size=5000, input_height=88, input_width=80, input_channels=1,
                 conv_n_maps=[32, 64, 64],
                 conv_kernel_sizes=[(8, 8), (4, 4), (3, 3)], conv_strides=[4, 2, 1],
                 n_hidden=512,
                 eps_min=0.1, eps_max=1.0, eps_decay_steps=2000000, learning_rate=.00025,
                 n_steps=4000000, training_start=10000, training_interval=16,
                 gamma=0.98, skip_start=90, batch_size=64, option='DDQN'):

        self.option = option

        # Reduced image size for DQN processing
        self.input_height = input_height  # How many pixels in height
        self.input_width = input_width  # How many pixels in width
        self.input_channels = input_channels  # How many color channels (if 1, set to gray scale)

        # NN hyperparameters
        self.conv_n_maps = conv_n_maps
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_strides = conv_strides
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.online_model = None
        self.target_model = None

        self.old_reward = None
        self.new_reward = None

        self.gamma = gamma  # the discount factor
        self.action = None

        self.batch_size = batch_size
        self.training_interval = training_interval

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

        self.learning_rate = learning_rate

        self.n_steps = n_steps  # total number of training steps
        self.skip_start = skip_start  # Skip the start of every game (it's just waiting time).
        self.done = True  # env needs to be reset
        self.training_start = training_start  # start training after 10,000 game iterations
        self.reward = 0

        self.create_dqns()

    def create_dqns(self):
        """Creates DQNs and connects networks via copy online to target procedure."""
        # create model
        self.target_model = Sequential()
        # add model layers
        self.target_model.add(Conv2D(self.conv_n_maps[0], kernel_size=self.conv_kernel_sizes[0],
                                     activation='relu', strides=self.conv_strides[0], data_format="channels_last",
                                     input_shape=(self.input_height, self.input_width, self.input_channels)))
        self.target_model.add(Conv2D(self.conv_n_maps[1], kernel_size=self.conv_kernel_sizes[1],
                                     activation='relu', strides=self.conv_strides[1], data_format="channels_last"))
        self.target_model.add(Conv2D(self.conv_n_maps[2], kernel_size=self.conv_kernel_sizes[2],
                                     activation='relu', strides=self.conv_strides[2], data_format="channels_last"))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(self.n_hidden, activation='relu'))
        self.target_model.add(Dense(self.n_outputs, activation='linear'))  # our computed q-values per action
        self.target_model.compile(optimizer=Adam(lr=self.learning_rate), metrics=['mae'], loss='mse')

        self.online_model = Sequential()
        # add model layers
        self.online_model.add(Conv2D(self.conv_n_maps[0], kernel_size=self.conv_kernel_sizes[0],
                                     activation='relu', strides=self.conv_strides[0], data_format="channels_last",
                                     input_shape=(self.input_height, self.input_width, self.input_channels)))
        self.online_model.add(Conv2D(self.conv_n_maps[1], kernel_size=self.conv_kernel_sizes[1],
                                     activation='relu', strides=self.conv_strides[1], data_format="channels_last"))
        self.online_model.add(Conv2D(self.conv_n_maps[2], kernel_size=self.conv_kernel_sizes[2],
                                     activation='relu', strides=self.conv_strides[2], data_format="channels_last"))
        self.online_model.add(Flatten())
        self.online_model.add(Dense(self.n_hidden, activation='relu'))
        self.online_model.add(Dense(self.n_outputs, activation='linear'))  # our computed q-values per action
        self.online_model.compile(optimizer=Adam(lr=self.learning_rate), metrics=['mae'], loss='mse')

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

        dead = False  # End game indicator (after 3 lives)
        game_reward = 0  # Game reward
        step = 0  # How many steps per episode (one live)
        lives = 3  # MsPacman lives

        try:
            if self.option == 'DQN':
                self.target_model = load_model('target_model_DQN')
            elif self.option == 'DDQN':
                self.target_model = load_model('target_model_DDQN')
                self.online_model = load_model('target_model_DDQN')
            print('Models loaded.')

        except:
            print('Cannot find any model to load.')

        for iteration in range(self.n_steps):

            if dead:
                print('Step: {}'.format(step))
                step = 0

            if self.done:  # episode over, start again
                self.reward = 0
                print("End Game reward: {}     Iteration: {}".format(game_reward, iteration))
                with open('games.csv', 'a')as f:
                    f.write('{},\n'.format(game_reward))
                game_reward = 0
                lives = 3
                obs = env.reset()
                print('MsPacman is dead, environment reset!')
                for skip in range(self.skip_start):  # skip the start of each game
                    obs, reward, self.done, info = env.step(0)
                state = self.preprocess_observation(obs)
                old_state = deepcopy(state)

            if iteration % 4 == 0:  # Make a decision only every 4 frames of game
                # Target DQN evaluates what to do (we use epsilon greedy here to explore an environment)
                target_q_values = self.target_model.predict((state - old_state).reshape(1, 88, 80, 1))
                self.action = self.epsilon_greedy(target_q_values, step)
                # Target DQN plays
                obs, reward, self.done, info = env.step(self.action)
                # print('End game: {}'.format(self.done))
                # time.sleep(0.5)
                dead = info['ale.lives'] < lives
                # print('Dead: {}'.format(dead))
                lives = info['ale.lives']
                # if an action make the Pacman dead, then gives penalty of -100
                reward = reward if not dead else -100
                game_reward += reward
                self.reward += reward
                # Show MsPacman
                env.render()
                # preprocess next state
                next_state = self.preprocess_observation(obs)
                # Store (s,a,r,s') into replay memory
                self.replay_memory.append((state - old_state, self.action, self.reward, next_state - state, self.done))
                old_state = deepcopy(state)
                state = deepcopy(next_state)

                if iteration < self.training_start:
                    continue

                else:
                    if self.option == 'DDQN' and iteration % 512 == 0:
                        self.target_model.set_weights(self.online_model.get_weights())
                        print('Online to target copied!')

                    if iteration % 2000 == 0:
                        if self.option == 'DQN':
                            self.target_model.save('target_model_DQN')
                        elif self.option == 'DDQN':
                            self.target_model.save('target_model_DDQN')
                        print('Model saved.')

                    # Sample random replays
                    state_val, action_val, rewards_val, next_state_val, continues_val = (
                        self.sample_memories(self.batch_size))

                    # compute online q-values and target q-values
                    if self.option == 'DQN':
                        target = self.target_model.predict(state_val)
                        target_validate = self.target_model.predict(next_state_val)
                        for i in range(self.batch_size):
                            if continues_val[i]:
                                target[i][action_val[i]] = rewards_val[i]
                            else:
                                target[i][action_val[i]] = rewards_val[i] + self.gamma * np.max(target_validate[i])
                        # print('Fit time.')
                        self.target_model.fit(state_val, target, batch_size=self.batch_size, epochs=1, verbose=0)

                    elif self.option == 'DDQN':
                        target = self.online_model.predict(state_val)
                        target_validate = self.target_model.predict(next_state_val)
                        for i in range(self.batch_size):
                            if continues_val[i]:
                                target[i][action_val[i]] = rewards_val[i]
                            else:
                                target[i][action_val[i]] = rewards_val[i] + self.gamma * np.max(target_validate[i])
                        self.online_model.fit(state_val, target, batch_size=self.batch_size, epochs=1, verbose=0)

            else:
                obs_, reward_, done_, info = env.step(0)
                dead = info['ale.lives'] < lives
                lives = info['ale.lives']
                reward_ = reward_ if not dead else -100
                game_reward += reward_
                self.reward += reward_
                env.render()

            step += 1

    @staticmethod
    def preprocess_observation(obs):
        """Scale and preprocess observation to gray scale."""
        img = obs[1:176:2, ::2]  # crop and downsize
        img = img.mean(axis=2)  # to grayscale
        img[img == _CONTRAST_] = 0  # improve contrast
        img = (img - 128) / 128 - 1  # normalize from -1. to 1.
        return img.reshape(88, 80, 1)


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    env.seed(123)
    pacman = MsPacman(n_outputs=9)
    pacman.explore(env=env)
