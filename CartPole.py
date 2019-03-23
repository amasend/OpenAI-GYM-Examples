import gym
import time
import numpy as np
import argparse


class BasicAgent:
    """This basic agent tries to keep the pole upright as long as this is possible.
    The environment is taken from OpenAI GYM "CartPole-v0"
    """
    def __init__(self, env, epochs=50, steps=1000, policy='basic_policy'):
        self.rewards = list()
        self.steps_number = list()
        self.step_time = list()
        self.epochs = epochs
        self.steps = steps
        self.policy = policy
        self.env = env

    @staticmethod
    def basic_policy(obs):
        """Basic policy for CartPole example:
        If an angle is negative (pole tilted to the left) move it to the left.
        If an angle is positive (pole tilted to the right) move it to the right."""
        angle = obs[2]
        return 0 if angle < 0 else 1

    @staticmethod
    def ang_vel_policy(obs):
        """Basic policy for CartPole example:
        If an angle is negative (pole tilted to the left) look at angular velocity.
        If an angle is positive (pole tilted to the right) look at angular velocity."""
        angle = obs[2]
        ang_velocity = obs[3]
        if angle < 0:
            if ang_velocity > 0:
                return 1
            else:
                return 0
        else:
            if ang_velocity > 0:
                return 0
            else:
                return 1

    def play(self):
        """Play method, runs whole experiment."""
        # Run an example X times to get average score (X = number of epochs)
        for epoch in range(self.epochs):
            # Initiate an agent (reset an environment and reward on each epoch)
            obs = self.env.reset()
            episode_reward = 0
            start_step_time = time.time()
            # Try to keep pool upright Y times (Y = number of steps)
            for step in range(self.steps):
                self.env.render()
                # Make a decision what to do next based on our basic policy
                if self.policy == 'basic_policy':
                    obs, reward, done, info = env.step(self.basic_policy(obs))
                elif self.policy == 'ang_vel_policy':
                    obs, reward, done, info = env.step(self.ang_vel_policy(obs))
                if done:
                    self.rewards.append(episode_reward)
                    self.steps_number.append(step)
                    break
                else:
                    episode_reward += reward
            end_step_time = time.time()
            self.step_time.append(end_step_time-start_step_time)

    def print_stats(self):
        """Prints whole statistics after experiment."""
        print('Average reward {} after {} epochs.'.format(np.mean(self.rewards), self.epochs))
        print('Reward STD: {} after {} epochs.'.format(np.std(self.rewards), self.epochs))
        print('Max reward {}'.format(np.max(self.rewards)))
        print('Min reward {}'.format(np.min(self.rewards)))
        print('Average steps number {} after {} epochs.'.format(np.mean(self.steps_number), self.epochs))
        print('Steps number STD: {} after {} epochs.'.format(np.std(self.steps_number), self.epochs))
        print('Max steps {}'.format(np.max(self.steps_number)))
        print('Min steps {}'.format(np.min(self.steps_number)))
        print('Average epoch time: {} sec std: {} sec'.format(np.mean(self.step_time), np.std(self.step_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""OpenAI CartPole toy example.
    This basic agent tries to keep the pole upright as long as this is possible.
    The environment is taken from OpenAI GYM "CartPole-v0""")
    parser.add_argument('--policy', type=str, help="""Type of policy to implement 
    in a CartPole toy example.
    ang_vel_policy - If an angle is negative (pole tilted to the left) look at angular velocity.
                If an angle is positive (pole tilted to the right) look at angular velocity.
    basic_policy - if an angle is negative (pole tilted to the left) move it to the left.
                If an angle is positive (pole tilted to the right) move it to the right.""", default='ang_vel_policy',
                        choices=['ang_vel_policy', 'basic_policy'])
    parser.add_argument('--epochs', type=int, help='Number of epochs.', default=10)
    parser.add_argument('--steps', type=int, help='Maximum number of steps per epoch.', default=1000)
    args = parser.parse_args()

    environment = gym.make("CartPole-v0")
    agent = BasicAgent(environment, args.epochs, args.steps, args.policy)
    agent.play()
    agent.print_stats()

    environment.close()
