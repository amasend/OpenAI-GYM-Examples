import pandas as pd
import matplotlib.pyplot as plt

# show policy-gradient improvement
data = pd.read_csv('policy_gradient_rewards.csv', index_col=0)

plt.fill_between(data.index.values, (data['mean'] - data['std']).values,
                (data['mean'] + data['std']).values, interpolate=True, alpha=0.4)
plt.plot(data.index.values, data['mean'].values, 'r')
plt.title('Average reward per game. \n(CartPole example with policy-gradient)')
plt.ylabel('Reward')
plt.xlabel('Game')
plt.show()