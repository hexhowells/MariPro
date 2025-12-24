import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ale_py
import gymnasium as gym
from model import ActorCritic, Encoder, ForwardModel, InverseModel
from utils import make_env, get_local_ip

from slate_agent import  SlateAgentICM
from slate import SlateClient

import threading


def train(
        env_id="SuperMarioBros-v0",
        total_updates=500_000,
        num_envs=8,
        rollout_len=5,
        gamma=0.99,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        icm_beta=0.2,          # inverse vs forward weighting
        icm_eta=0.01,          # scale intrinsic reward
        icm_coef=1.0,          # weight icm loss in total loss
        seed=0,
    ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i) for i in range(num_envs)])
    obs, _ = envs.reset()
    
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(actions=act_dim).to(device)
    encoder_model = Encoder(actions=act_dim).to(device)
    forward_model = ForwardModel(actions=act_dim).to(device)
    inverse_model = InverseModel(actions=act_dim).to(device)

    inverse_loss = nn.CrossEntropyLoss()
    forward_loss = nn.MSELoss()

    optimizer = optim.Adam(
        list(model.parameters())
        + list(encoder_model.parameters())
        + list(forward_model.parameters())
        + list(inverse_model.parameters()),
        lr=lr,
    )

    # run slate
    def run_client():
        env = make_env(env_id, 1)()
        agent = SlateAgentICM(env, 'checkpoints')
        runner = SlateClient(
            env, 
            agent, 
            endpoint=get_local_ip(), 
            run_local=True, 
            checkpoints_dir='checkpoints',
            )
        runner.start_client()
    
    thread = threading.Thread(target=run_client, daemon=True)
    thread.start()

    for update in range(1, total_updates + 1):
        obs_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []
        icm_loss_buf = []

        prev_score = np.zeros(envs.num_envs)

        for t in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits, values = model(obs_t)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            # compute reward with ICM
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            phi_s = encoder_model(obs_t)
            phi_next = encoder_model(next_obs_t)

            inv_logits = inverse_model(torch.cat([phi_s, phi_next], dim=1))
            inv_loss = inverse_loss(inv_logits, actions)

            actions_one_hot = F.one_hot(actions, act_dim).float()
            phi_pred = forward_model(torch.cat([phi_s, actions_one_hot], dim=1))
            fwd_loss = forward_loss(phi_pred, phi_next.detach())

            icm_loss = ((1 - icm_beta) * inv_loss) + (icm_beta * fwd_loss)

            intrinsic_reward = icm_eta * 0.5 * (phi_pred.detach() - phi_next.detach()).pow(2).sum(dim=1)

            total_reward = intrinsic_reward

            obs_buf.append(obs_t)
            logp_buf.append(logp)
            rew_buf.append(total_reward)
            done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            val_buf.append(values)
            icm_loss_buf.append(icm_loss)

            obs = next_obs

        # compute value of last state
        with torch.no_grad():
            obs_last = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_values = model(obs_last)

        # n-step rewards
        returns = []
        R = last_values
        for t in reversed(range(rollout_len)):
            R = rew_buf[t] + (gamma * R * (1.0 - done_buf[t]))  # if done, R = rew_buf[t]
            returns.append(R)

        returns.reverse()

        # flatten time and env dims to make mini-batch
        T, N = rollout_len, num_envs
        returns = torch.stack(returns).reshape(T * N)
        values = torch.stack(val_buf).reshape(T * N) 
        logps = torch.stack(logp_buf).reshape(T * N) 
        obs_stack = torch.stack(obs_buf).reshape(T * N, *obs_dim)
        icm_loss_total = torch.stack(icm_loss_buf).mean()

        advantages = returns - values

        policy_loss = -(logps * advantages.detach()).mean()

        value_loss = 0.5 * (returns - values).pow(2).mean()

        logits_flat, _ = model(obs_stack)
        dist_flat = torch.distributions.Categorical(logits=logits_flat)
        entropy_loss = -dist_flat.entropy().mean()

        loss = policy_loss + vf_coef * value_loss + (ent_coef * entropy_loss) + icm_coef * icm_loss_total

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if update % 50 == 0:
            with torch.no_grad():
                v_mean = last_values.mean().item()
            print(
                f"update={update:4d} "
                f"loss={loss.item():.3f} "
                f"pi={policy_loss.item():.3f} "
                f"vf={value_loss.item():.3f} "
                f"ent={(-entropy_loss).item():.3f} "
                f"icm={icm_loss_total.item():.3f} "
                f"Vmean={v_mean:.3f}"
            )
        
        if update % 500 == 0:
            checkpoint_path = f"checkpoints/model_{update:07}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    envs.close()
    return model


if __name__ == "__main__":
    train()
