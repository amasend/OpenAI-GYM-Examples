import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


class SimpleExample:
    """Class for simple q-learning example."""
    def __init__(self, alpha=0.8, gamma=0.99, ep_number=100):
        self.game_board = pd.DataFrame(data=np.array([[0, 0, 0, 1],
                                                      [0, 2, 0, 1],
                                                      [0, 1, 2, 0],
                                                      [2, 0, 0, 3]]),
                                       columns=['a', 'b', 'c', 'd'],
                                       index=[1, 2, 3, 4])
        self.reward_table = pd.DataFrame(data=np.array(
            [[np.nan, -1, -1, np.nan, np.nan, np.nan, -100, 1, np.nan, -1, np.nan, -100, np.nan, 1, 1, np.nan],
             [-1, -1, -100, np.nan, -100, np.nan, -1, np.nan, -1, -100, np.nan, np.nan, 1, -1, 100, np.nan],
             [-1, -100, 1, np.nan, -1, np.nan, -100, -1, 1, 1, np.nan, 100, np.nan, np.nan, np.nan, np.nan],
             [np.nan, np.nan, np.nan, np.nan, -1, np.nan, -1, -100, -1, -100, np.nan, -1, -1, -1, -100, np.nan]
             ]),
                                         columns=['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2', 'c3',
                                                  'c4', 'd1', 'd2', 'd3', 'd4'],
                                         index=['up', 'down', 'right', 'left'])
        self.q_table = pd.DataFrame(data=np.zeros((4, 16)),
                                    columns=['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2', 'c3', 'c4',
                                             'd1', 'd2', 'd3', 'd4'],
                                    index=['up', 'down', 'right', 'left'])
        self.all_states = ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4', 'c1', 'c2', 'c3', 'c4',
                           'd1', 'd2', 'd3', 'd4']
        self.all_actions = ['up', 'down', 'right', 'left']
        self.current_state = ''
        self.current_action = ''
        self.learning_rate = alpha
        self.discount_factor = gamma
        self.number_of_episodes = ep_number
        self.terminate = False
        self.reward = None
        # self.fig = plt.figure()
        # self.timer = self.fig.canvas.new_timer(interval=2000)
        # self.timer.add_callback(self.close_event)

    def run_bellman_equation(self, next_state=None, terminate=False):
        """Compute Bellman equation for current state and update current q-value"""
        if terminate:
            self.q_table.loc[self.current_action, self.current_state] = (1 - self.learning_rate) * \
                self.q_table.loc[self.current_action, self.current_state] + self.learning_rate * self.reward
        else:
            self.q_table.loc[self.current_action, self.current_state] = (1 - self.learning_rate) * \
                self.q_table.loc[self.current_action, self.current_state] + self.learning_rate * \
                (self.reward + self.discount_factor * np.max(self.q_table.loc[:, next_state]))

    def map_action_to_next_state(self):
        """Maps action to the next state in string."""
        if self.current_action == 'up':
            return self.current_state[0] + str(int(self.current_state[1]) - 1)
        elif self.current_action == 'down':
            return self.current_state[0] + str(int(self.current_state[1]) + 1)
        elif self.current_action == 'right':
            return chr(ord(self.current_state[0]) + 1) + self.current_state[1]
        elif self.current_action == 'left':
            return chr(ord(self.current_state[0]) - 1) + self.current_state[1]

    def explore_environment(self):
        """Explore the environment by agent."""
        for i in range(self.number_of_episodes):  # For each episone
            print('Episode {}'.format(i))
            self.terminate = False
            while True:  # Choose randoom state and action (only available)
                self.current_action = np.random.choice(self.all_actions, 1)[0]
                self.current_state = np.random.choice(self.all_states, 1)[0]
                self.reward = self.reward_table.loc[self.current_action, self.current_state]  # Compute reward
                if np.isnan(self.reward):
                    continue
                elif self.reward == 100 or self.reward == -100:  # If we choose the ending state/action
                    # time.sleep(2)
                    # Compute Bellman equation without future reward (only +/-100 for the end game)
                    self.run_bellman_equation(terminate=True)
                    self.terminate = True
                    break
                else:
                    break
            print('State: {}    action: {}'.format(self.current_state, self.current_action))
            while not self.terminate:
                # time.sleep(2)
                next_state = self.map_action_to_next_state()  # What is the next state?
                # Compute whole Bellman equation with discounted q-value reward
                self.run_bellman_equation(next_state=next_state)
                self.current_state = next_state  # Switch to the next state
                while True:
                    self.current_action = np.random.choice(self.all_actions, 1)[0]  # Choose next random action
                    self.reward = self.reward_table.loc[self.current_action, self.current_state]
                    # State/action should be available
                    if np.isnan(self.reward):
                        continue
                    elif self.reward == 100 or self.reward == -100:
                        # time.sleep(2)
                        self.run_bellman_equation(terminate=True)
                        self.terminate = True
                        break
                    else:
                        break
                print('State: {}    action: {}'.format(self.current_state, self.current_action))
        self.q_table.to_csv('q_table.csv')

    def run_game(self):
        """Here you can run a agent to exploit the environment."""
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nGame phase')
        self.current_state = 'a1'  # Define start position
        while True:
            self.current_action = self.q_table[self.current_state].idxmax()  # Choose action based on the max(q-value)
            print('State: {}    action: {}'.format(self.current_state, self.current_action))
            self.reward = self.reward_table.loc[self.current_action, self.current_state]
            if self.reward == 100:
                break
            self.current_state = self.map_action_to_next_state()

    @staticmethod
    def close_event():
        plt.close()  # timer calls this function after 3 seconds and closes the window


x = SimpleExample()
x.explore_environment()
x.run_game()
