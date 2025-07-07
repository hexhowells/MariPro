# Deep Q-Network

[Deep Q-Network](https://www.nature.com/articles/nature14236) is an expansion on reinforcement learning algorithm [Q-learning](https://en.wikipedia.org/wiki/Q-learning), in which the q-value function is approximated using neural networks.

Here, you have an agent which acts within an environment using a discrete set of actions taken at timesteps, certain states within the environment produce reward values in which the agent is trying to maximise.

Q-learning is a value-based reinforcement learning algorithm in which a Q-value function (poilcy network) is given a state (s) and action (a) pair, and learns to predict the reward of taking the action in the given state. For DQN, this can be formally written as: (where `θ` are the parameters of a neural network)
```math
\large{Q = (s, a; θ)}
```

Typically the model used for DQN is a convolutional network to process image data, which is then fed into some linear layers to produce rewards for each potential action that can be taken. The loss function of this network is defined with the following:

## Loss function
```math
Loss =  \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)^2
```
Where:

`r` = reward

`γ` = discount factor

`max Q(s', a')` = maximum predicted reward for the next state using the target network

`Q(s, a)` = predicted reward for the action taken by the current policy network

## Target network and discount factor
Here the target network is a copy of the policy network but updated less frequently (its never trained, just the weights are copied from the policy net). This acts as a stabiliser so the policy network is free to explore outputs whilst the target network produces more stable reward predictions. The combination of a target network and discount factor is to resolve the issue of sparse rewards, in which not every state produces a positive reward signal. Since the target network predicts the reward of the *next* state, states, it will predict positive reward signals for state-action pairs leading up to a reward. The discount factor is multiplied with the target networks's predictions to taper each prediction down. As such a discount factor closer to 1 (0.99) enables the network to distill rewards backwards in time for longer periods.

For a visual, assume a state a t=5 recieved a positive reward of `+1`, here are how the rewards would be calcuated for each state from t=5 to t=1:

```
t=5  Q(s5, a5) = (r=1) + 0.9 * QT(s6)=0  = (1 + 0.9 * 0) = 1
t=4  Q(s4, a4) = (r=0) + 0.9 * QT(s5)=1  = (0 + 0.9 * 1) = 0.9
t=3  Q(s3, a3) = (r=0) + 0.9 * QT(s4)=0.9  = (0 + 0.9 * 0.9) = 0.81
t=2  Q(s2, a2) = (r=0) + 0.9 * QT(s3)=0.81  = (0 + 0.9 * 0.81) = 0.729
t=1  Q(s1, a1) = (r=0) + 0.9 * QT(s2)=0.729  = (0 + 0.9 * 0.729) = 0.6561
```
Where QT(st) is the target network predicting the value of state `s` at timestep `t`. We can see that the positive reward at timestep 5 get distilled down to the states leading up to the state via the target networks predictions. The above assumes the target network is able to perfectly predict the max reward of each state and that the appropriate actions are taken at each timestep to maximise the reward.

## Experience replay
The policy network can make observations and take actions in the environment, collect each state transition (state, action, next_state, reward) in a batch, and perform backpropagation on that data. However, doing this would introduce a lot of correlation in the training data, as each minibatch is highly correlated to each other, and this be quite unstable. In order to combat this, we can use experiance replay buffers.

Experience replay buffers store the last N transitions (e.g 1,000,000 transitions) in a buffer. Then for each training minibatch, a minibatch of transitions can be uniformly sampled from the replay buffer, introducing a lot more diveristy in the minibatch and greatly stabilises training.


# Double DQN

# Prioritised Experience Replay
