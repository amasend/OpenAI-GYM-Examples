import gym
import time
import numpy as np


def basic_policy(obs):
    """Basic policy for CartPole example:
    If an angle is negative (pole tilted to the left) move it to the left.
    If an angle is positive (pole tilted to the right) move it to the right."""
    angle = obs[2]
    return 0 if angle < 0 else 1


def basic_agent(env, epochs=50, steps=1000):
    """This basic agent tries to keep the pole upright as long as this is possible.
    The environment is taken from OpenAI GYM "CartPole-v0"
    """
    rewards = list()
    steps_number = list()
    step_time = list()
    # Run an example X times to get average score (X = number of epochs)
    for epoch in range(epochs):
        # Initiate an agent (reset an environment and reward on each epoch)
        obs = env.reset()
        episode_reward = 0
        start_step_time = time.time()
        # Try to keep pool upright Y times (Y = number of steps)
        for step in range(steps):
            env.render()
            # Make a decision what to do next based on our basic policy
            obs, reward, done, info = env.step(basic_policy(obs))
            if done:
                rewards.append(episode_reward)
                steps_number.append(step)
                break
            else:
                episode_reward += reward
        end_step_time = time.time()
        step_time.append(end_step_time-start_step_time)

    print('Average reward {} after {} epochs.'.format(np.mean(rewards), epochs))
    print('Reward STD: {} after {} epochs.'.format(np.std(rewards), epochs))
    print('Max reward {}'.format(np.max(rewards)))
    print('Min reward {}'.format(np.min(rewards)))
    print('Average steps number {} after {} epochs.'.format(np.mean(steps_number), epochs))
    print('Steps number STD: {} after {} epochs.'.format(np.std(steps_number), epochs))
    print('Max steps {}'.format(np.max(steps_number)))
    print('Min steps {}'.format(np.min(steps_number)))
    print('Average epoch time: {} sec std: {} sec'.format(np.mean(step_time), np.std(step_time)))


env = gym.make("CartPole-v0")
basic_agent(env, 10, 1000)

env.close()
