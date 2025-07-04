from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
#import ale_py
#import gymnasium as gym

import wandb


from collections import deque

import torch
import torchvision.transforms as T

from model import QNetwork
from replay_buffer import PrioritisedReplayBuffer
from env import Environment, Breakout
from utils import epsilon_greedy, compute_targets, evaluate
import hyperparameters as hp

from tqdm import tqdm


# create environment
_env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
_env = JoypadSpace(_env, SIMPLE_MOVEMENT)
env = Environment(_env)

_env_eval = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="rgb_array")
_env_eval = JoypadSpace(_env_eval, SIMPLE_MOVEMENT)
env_eval = Environment(_env_eval)


# gym.register_envs(ale_py)
# _env = gym.make("ALE/Breakout-v5")
# env = Breakout(_env)

# _env_eval = gym.make("ALE/Breakout-v5")
# env_eval = Breakout(_env)

hp.action_space = _env.action_space.n

# create policy and target networks
policy_net = QNetwork(hp.in_channels, hp.action_space).to(device='cuda')
target_net = QNetwork(hp.in_channels, hp.action_space).to(device='cuda')
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# set hyperparameters
criterion = torch.nn.SmoothL1Loss()  # Huber loss
optimiser = torch.optim.Adam(policy_net.parameters(), lr=hp.lr, eps=1e-4)

transform = T.Compose([
    T.ToPILImage(),
    T.Grayscale(1),
    T.Resize((84, 84)),
    T.ToTensor()
])

steps = 0
best_score = 0
epsilon = hp.epsilon
replay_buffer = PrioritisedReplayBuffer(
    capacity=hp.replay_memory_size,
    frame_stack=hp.history_len,
    transform=transform
    )

eval_score = evaluate(env_eval, policy_net, transform)
episode = 1

pbar = tqdm(total = hp.total_steps)

wandb.init(
    project="mario-dqn",
    config={k: v for k, v in hp.__dict__.items() if not k.startswith("__")},
)


while steps <= hp.total_steps:
    episode += 1

    # reset environment and get first state
    first_frame, *_ = env.reset()
    replay_buffer.add_first_frame(first_frame)

    done = False
    losses = []

    while not done:
        steps += 1

        # create state
        state = replay_buffer.current_state()
        
        # sample the action with epsilon-greedy
        action = epsilon_greedy(policy_net, state, epsilon)

        # execute action
        next_frame, reward, done = env.step(action)

        # store transition in buffer
        replay_buffer.append(next_frame, action, reward, done, priority=1.0)

        # dont train until replay buffer fits a single batch
        if len(replay_buffer) <= hp.replay_start_size:
            episode = 0
            steps = 0
            continue

        pbar.update(1)

        # update q-network
        if (steps % hp.update_frequency) == 0:
            #states, actions, targets, weights, indices = create_minibatch(replay_buffer, policy_net, target_net)
            *transitions, weights, indices = replay_buffer.sample(hp.batch_size)
            states, actions, _, _, _ = transitions
            targets = compute_targets(policy_net, target_net, transitions)

            q_values = policy_net(states.to(device='cuda')).gather(1, actions).squeeze()

            td_errors = (targets - q_values).abs()
            replay_buffer.update_priorities(indices, td_errors.detach().cpu())

            loss = criterion(q_values, targets)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(loss.detach().item())

        # update target network
        if (steps % hp.target_update_frequency) == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (steps % hp.eval_steps) == 0:
            eval_score = evaluate(env_eval, policy_net, transform)

        # decay epsilon
        if epsilon != hp.epsilon_min:
            epsilon = max(hp.epsilon_min, hp.epsilon - (hp.epsilon - hp.epsilon_min) * (steps / hp.epsilon_decay_steps))

        # save model
        if (steps % hp.checkpoint_steps) == 0:
            torch.save(policy_net.state_dict(), f"models/model_{steps}.pth")

    if len(losses) > 0:
        wandb.log({
            "episode": episode,
            "average_loss": sum(losses) / len(losses),
            "total_reward": env.total_reward,
            "epsilon": epsilon,
            "eval_score": eval_score,
            "steps": steps,
        }, step=steps)
