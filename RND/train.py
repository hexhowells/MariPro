import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ale_py
import gymnasium as gym
from model import ActorCritic, TargetNetwork, PredictionNetwork
from utils import make_env, get_local_ip, RunningMeanStd

from slate_agent import SlateAgentRND
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
        rnd_eta=0.01,          # scale intrinsic reward
        rnd_coef=1.0,          # weight rnd loss in total loss
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
    target_model = TargetNetwork().to(device)
    target_model.eval()
    prediction_model = PredictionNetwork().to(device)

    for param in target_model.parameters():
        param.requires_grad = False

    rnd_loss_fn = nn.MSELoss(reduction='none')
    
    int_reward_rms = RunningMeanStd()

    optimizer = optim.Adam(
        list(model.parameters()) + list(prediction_model.parameters()),
        lr=lr,
    )

    # run slate
    def run_client():
        env = make_env(env_id, 1)()
        agent = SlateAgentRND(env, 'checkpoints')
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
        rnd_loss_buf = []
        intrinsic_rewards = []

        for t in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits, values = model(obs_t)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            # Compute RND intrinsic reward
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                target_features = target_model(next_obs_t)
            
            pred_features = prediction_model(next_obs_t)
            rnd_loss = rnd_loss_fn(pred_features, target_features).mean(dim=1)
            
            raw_intrinsic_reward = rnd_loss.detach()
            
            normalized_int_reward_np = int_reward_rms.normalize(raw_intrinsic_reward)
            normalized_int_reward = torch.tensor(
                normalized_int_reward_np,
                dtype=torch.float32,
                device=device
            )
            intrinsic_reward = rnd_eta * normalized_int_reward
            intrinsic_rewards.append(intrinsic_reward)
            
            extrinsic_reward = torch.tensor(rewards, dtype=torch.float32, device=device)
            total_reward = extrinsic_reward + intrinsic_reward

            obs_buf.append(obs_t)
            logp_buf.append(logp)
            rew_buf.append(total_reward)
            done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            val_buf.append(values)
            rnd_loss_buf.append(rnd_loss.mean())

            obs = next_obs
        
        all_raw_int_rewards = torch.cat(intrinsic_rewards)
        int_reward_rms.update(all_raw_int_rewards)

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
        rnd_loss_total = torch.stack(rnd_loss_buf).mean()

        advantages = returns - values

        policy_loss = -(logps * advantages.detach()).mean()

        value_loss = 0.5 * (returns - values).pow(2).mean()

        logits_flat, _ = model(obs_stack)
        dist_flat = torch.distributions.Categorical(logits=logits_flat)
        entropy_loss = -dist_flat.entropy().mean()

        loss = policy_loss + vf_coef * value_loss + (ent_coef * entropy_loss) + rnd_coef * rnd_loss_total

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(prediction_model.parameters()), 
            max_grad_norm
        )
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
                f"rnd={rnd_loss_total.item():.3f} "
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
