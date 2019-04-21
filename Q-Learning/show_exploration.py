import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from multiprocessing.connection import Listener


def map_action_to_next_state(state, action):
    """Maps action to the next state in string."""
    if action == 'up':
        return state[0] + str(int(state[1]) - 1)
    elif action == 'down':
        return state[0] + str(int(state[1]) + 1)
    elif action == 'right':
        return chr(ord(state[0]) + 1) + state[1]
    elif action == 'left':
        return chr(ord(state[0]) - 1) + state[1]


def exploration():
    """Plot q-table and game board for reference during exploration."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plt.ion()
    plt.show()
    address = ('localhost', 6000)  # family is deduced to be 'AF_INET'
    listener = Listener(address, authkey=b'secret password')
    conn = listener.accept()
    print('connection accepted from {}'.format(listener.last_accepted))

    while True:
        msg = conn.recv()
        ax1.clear()
        ax2.clear()
        q_table = msg['q_table']
        game_board = msg['game_board']
        cords = msg['cords']
        action = msg['action']

        next_state = map_action_to_next_state(cords, action)
        print('State: {}    action: {}    next {}'.format(cords, action, next_state))
        next_row = int(next_state[1])
        next_col = next_state[0]
        row = int(cords[1])
        col = cords[0]

        game_board.loc[row, col] = -3  # Mark current state on a game board
        game_board.loc[next_row, next_col] = -2  # mark next state on a game board

        # Plot q-table and game board as heatmaps
        sns.heatmap(q_table, ax=ax1, cbar=False, vmin=-200, vmax=200, annot=True)
        sns.heatmap(game_board, ax=ax2, cbar=False, vmin=-3, vmax=3, annot=True)
        plt.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    exploration()
