import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium
from model import ActorCritic
from utils import make_env, render_policy


def train(
        env_id="SuperMarioBros-v0",
        total_updates=100_000,
        num_envs=8,
        rollout_len=5,
        gamma=0.99,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=0,
    ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gymnasium.vector.SyncVectorEnv([make_env(env_id, seed + i) for i in range(num_envs)])
    obs, _ = envs.reset()
    
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(actions=act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for update in range(0, total_updates + 1):
        obs_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        for t in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            logits, values = model(obs_t)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            logp = dist.log_prob(actions)
            next_obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            obs_buf.append(obs_t)
            logp_buf.append(logp)
            rew_buf.append(torch.tensor(rewards, dtype=torch.float32, device=device))
            done_buf.append(torch.tensor(dones, dtype=torch.float32, device=device))
            val_buf.append(values)

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

        advantages = returns - values

        policy_loss = -(logps * advantages.detach()).mean()

        value_loss = 0.5 * (returns - values).pow(2).mean()

        logits_flat, _ = model(obs_stack)
        dist_flat = torch.distributions.Categorical(logits=logits_flat)
        entropy_loss = -dist_flat.entropy().mean()

        loss = policy_loss + vf_coef * value_loss + (ent_coef * entropy_loss)

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
                f"Vmean={v_mean:.3f}"
            )
        
        if update % 500 == 0:
            eval_return = render_policy(model, env_id=env_id)
            print(f'{eval_return=}')
        
        if update % 1000 == 0:
            checkpoint_path = f"{checkpoint_dir}/model_{update}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    envs.close()
    
    final_checkpoint = f"{checkpoint_dir}/model_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Saved final checkpoint: {final_checkpoint}")
    
    return model


if __name__ == "__main__":
    train()
