import gym
import time
import numpy as np
import argparse
import tensorflow as tf


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
        self.action = None
        self.gradients = list()
        self.training_op = None
        self.init = None
        self.saver = None
        self.gradient_placeholders = list()
        self.grads_and_vars_feed = list()

    @staticmethod
    def discount_rewards(rewards, gamma):
        """Computes discountet rewards.
        rewards - one single array contains all rewards from one game in order game([rew_1, rew_2, rew_3 ...])

        It applies a discount factor to more distant rewards. Basically you do not know what will be
        in the future for sure, so it is better to focus on more present actions and their rewards.
                gamma^0 * rew_1 + gamma^1 * rew_2 + gamma^2 * rew_3 ...

        Below algorithm applies discount factor in the following way:
                discounted_rewards = [ , , , , ...]
                discounted_rewards = [ , , , , ... , reward(n)]
                discounted_rewards = [ , , , , reward(n-1) + reward(n) * gamma, reward(n)]
                discounted_rewards = [reward(n-2) + reward(n-1) * gamma + reward(n) * gamma^2,
                                                    reward(n-1) + reward(n) * gamma, reward(n)]"""
        discounted_rewards = np.empty(len(rewards))
        cumulative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumulative_rewards = rewards[step] + cumulative_rewards * gamma
            discounted_rewards[step] = cumulative_rewards
        return discounted_rewards

    def discount_and_normalize_rewards(self, all_rewards, gamma):
        """Compute normalized discounted rewards.
        1. Compute mean and std of the all discounted rewards from all games. (by concatenate each game)
        2. Normalize each of the game discounted rewards by substracting the mean and dividing by std."""
        all_discounted_rewards = [self.discount_rewards(rewards, gamma) for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        reward_mean = flat_rewards.mean()
        reward_std = flat_rewards.std()
        return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

    def neural_net_model(self):
        """Creates tensorflow graph, neural network.
        Also defines:
        - tensorflow initialize method
        - tensorflow apply gradient method
        - tensorflow save method"""
        self.n_inputs = 4  # Number of input neurons, should be equal to the number of environment descriptive features
        n_hidden = 4  # Small problem requires small network (compute faster)
        n_outputs = 1  # Problem is a binary decision (0 or 1)
        initializer = tf.contrib.layers.variance_scaling_initializer()  # Initializer of the neuron weights

        learning_rate = 0.01  # Optimizer learning rate

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])  # Creates placeholder for the first layer
        hidden = tf.layers.dense(self.X, n_hidden, activation=tf.nn.elu,  # Creates first hidden layer with ELU activation
                                 kernel_initializer=initializer)  # function and initiate neuron weights
        logits = tf.layers.dense(hidden, n_outputs,  # Creates output layer with only one neuron (binary classification)
                                 kernel_initializer=initializer)
        outputs = tf.nn.sigmoid(logits)  # Applies the last activation function (sigmoid) to obtain probability
        # Outputs produces probability only for going for one direction eg. going left, for multinomial
        # function we need specify both probabilities (for going left and for going right) so concatenation should take
        # place here, concat(going_left_prob, going_right_prob)
        p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
        # Samples one integer 0 or 1 based on their probability, produces final action
        self.action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

        y = 1. - tf.to_float(self.action)  # Make target probability (if action is 0, target prob should be 1.)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y, logits=logits)  # Compute cross entropy
        optimizer = tf.train.AdamOptimizer(learning_rate)  # Initialize optimizer
        # Compute gradients (returns pairs (gradient, training variable)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)
        self.gradients = [grad for grad, variable in grads_and_vars]  # Obtain only gradients
        # Creates placeholders for gradient vectors and gradient plus variable vectors for further
        # updates
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            self.gradient_placeholders.append(gradient_placeholder)
            self.grads_and_vars_feed.append((gradient_placeholder, variable))

        self.training_op = optimizer.apply_gradients(self.grads_and_vars_feed)  # Applies tuned gradients to network
        self.init = tf.global_variables_initializer()  # Initialize all variables (weights etc.)
        self.saver = tf.train.Saver()  # Saves neural network state

    def play_neural_net(self):
        """Plays CartPole with Neural Network as an agent.
        It tries to run number of epochs with 10 games each, where one game tries to do number of steps.
        After each 10 games, save the model state."""
        n_iterations = 250  # number of training iterations
        n_max_steps = 1000  # max steps per episode
        n_games_per_update = 10  # train the policy every 10 episodes
        save_iterations = 10  # save the model every 10 training iterations
        gamma = 0.95  # the discount factor

        # Starts whole tensorflow session
        with tf.Session() as sess:
            self.init.run()  # Initialize all variables in NN
            for iteration in range(n_iterations):
                all_rewards = []  # all sequences of raw rewards for each episode
                all_gradients = []  # gradients saved at each step of each episode
                for game in range(n_games_per_update):
                    current_rewards = []  # all raw rewards from the current episode
                    current_gradients = []  # all gradients from the current episode
                    obs = self.env.reset()
                    for step in range(n_max_steps):
                        self.env.render()  # Show environment
                        # Computes action and gradients per step
                        action_val, gradients_val = sess.run([self.action, self.gradients],
                                                             feed_dict={self.X: obs.reshape(1, self.n_inputs)})
                        obs, reward, done, info = self.env.step(action_val[0][0])
                        # Updates arrays with rewards and gradients per step
                        current_rewards.append(reward)
                        current_gradients.append(gradients_val)
                        if done:
                            break
                    # Updates arrays with rewards and gradients per game
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)
                # At this point we have run the policy for 10 episodes, and we are
                # ready for a policy update with the following algorithm:
                # - Each computed gradient should be multiply by particular reward got along with this gradient.
                # - Then mean gradient value should be computed
                all_rewards = self.discount_and_normalize_rewards(all_rewards, gamma)
                feed_dict = {}
                for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
                    # multiply the gradients by the action scores, and compute the mean
                    mean_gradients = np.mean(
                        [reward * all_gradients[game_index][step][var_index]
                         for game_index, rewards in enumerate(all_rewards)
                         for step, reward in enumerate(rewards)],
                        axis=0)
                    feed_dict[grad_placeholder] = mean_gradients
                sess.run(self.training_op, feed_dict=feed_dict)  # Applies mean gradient to the network
                if iteration % save_iterations == 0:  # Save model state after each 10th iteration
                    self.saver.save(sess, ".\\my_policy_net_pg.ckpt")

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
        If angle velocity is negative move to the right, if not, move to the left.
        If an angle is positive (pole tilted to the right) look at angular velocity.
        If angle velocity is negative move to the left, if not, move to the right."""
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
                    obs, reward, done, info = self.env.step(self.basic_policy(obs))
                elif self.policy == 'ang_vel_policy':
                    obs, reward, done, info = self.env.step(self.ang_vel_policy(obs))
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
                If an angle is positive (pole tilted to the right) move it to the right.
    policy_gradient - based on neural network""", default='basic_policy',
                        choices=['ang_vel_policy', 'basic_policy', 'policy_gradient'])
    parser.add_argument('--epochs', type=int, help='Number of epochs.', default=10)
    parser.add_argument('--steps', type=int, help='Maximum number of steps per epoch.', default=1000)
    args = parser.parse_args()

    environment = gym.make("CartPole-v0")
    agent = BasicAgent(environment, args.epochs, args.steps, args.policy)
    if args.policy != 'policy_gradient':
        agent.play()
        agent.print_stats()
    else:
        agent.neural_net_model()
        agent.play_neural_net()

    environment.close()
