from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym

from collections import deque

import torch
import torchvision.transforms as T

from model import QNetwork
from frame_buffer import FrameBuffer
from env import Environment
from utils import epsilon_greedy, create_minibatch
import hyperparameters as hp



# create environment
_env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
_env = JoypadSpace(_env, SIMPLE_MOVEMENT)
env = Environment(_env)

# create policy and target networks
policy_net = QNetwork(hp.in_channels, hp.action_space).to(device='cuda')
target_net = QNetwork(hp.in_channels, hp.action_space).to(device='cuda')
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# set hyperparameters
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(policy_net.parameters(), lr=hp.lr)

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(1),
    T.Resize((84, 84)),
    T.ToTensor()
])

steps = 0
best_score = 0
replay_buffer = deque(maxlen=hp.replay_memory_size)


for episode in range(hp.num_episodes):
    print(f'\nStarting epidode: {episode}')

    # reset environment and get first state
    first_frame, *_ = env.reset()

    # store last N frames in state
    frame_stack = FrameBuffer(first_frame, hp.history_len, transform)
    state = frame_stack.state()

    done = False
    losses = []

    while not done:
        steps += 1
        # sample the action with epsilon-greedy
        action = epsilon_greedy(policy_net, state, hp.epsilon)

        # execute action
        next_frame, reward, done = env.step_n_times(action, hp.action_repeats)

        frame_stack.append(next_frame)

        # store transition in buffer
        replay_buffer.append((state, action, reward, frame_stack.state(), done))

        # dont train until replay buffer fits a single batch
        if len(replay_buffer) <= hp.batch_size:
            continue

        # update q-network
        if (steps % hp.update_frequency) == 0:
            states, actions, targets = create_minibatch(replay_buffer, target_net)
            q_values = policy_net(states.to(device='cuda')).cpu().gather(1, actions).squeeze()

            loss = criterion(q_values, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.detach().item())

        # update target network
        if (steps % hp.target_update_frequency) == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # decay epsilon
    if hp.epsilon > hp.epsilon_min:
        hp.epsilon *= hp.epsilon_decay

    if len(losses) > 0:
        print(f'  Average loss: {(sum(losses) / len(losses)):.2f}.')
        print(f'  Total reward: {env.total_reward:.2f}.  Current epsilon: {hp.epsilon:.2f}')
        print(f'  Max x-position: {env.high_score}')

        # save model on new high score
        if env.high_score > best_score:
            best_score = env.high_score
            torch.save(policy_net.state_dict(), f"models/model_{episode}.pth")
