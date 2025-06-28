import hyperparameters as hp
import random
import torch



def epsilon_greedy(policy_net, frame, epsilon):
    if random.random() < epsilon:
        action = random.randint(0, hp.action_space-1)
    else:
        with torch.no_grad():
            q_values = policy_net(frame.to(device='cuda'))
            action = torch.argmax(q_values).item()
    
    return action


def compute_reward(state, target_net):
    (frame, action, reward, next_frame, done) = state
    if done:
        return reward
    else:
        with torch.no_grad():
            target_q_value = target_net(next_frame.to(device='cuda')).max(1)[0]

        return reward + hp.discount_factor * target_q_value


def sample_buffer(replay_buffer, target_net):
    transitions = random.sample(replay_buffer, hp.batch_size)
    minibatch = []

    for transition in transitions:
        state = transition[0][0]  # extra [0] to remove an extra dimension
        action = transition[1]
        reward = compute_reward(transition, target_net)

        minibatch.append((state, action, reward))

    return minibatch


def create_minibatch(replay_buffer, target_net):
    minibatch = sample_buffer(replay_buffer, target_net)

    states, actions, targets = zip(*minibatch)

    states = torch.stack(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    targets = torch.Tensor(targets)

    return states, actions, targets
