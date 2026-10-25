import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import gymnasium as gym
from model import ActorCritic
from utils import make_env, render_policy, get_local_ip
import os

RUN_SLATE = bool(os.getenv('SLATE', False))

if RUN_SLATE:
    from slate_agent import SlateAgentPPO
    from slate import SlateClient
    import threading

torch.set_float32_matmul_precision('high')


def train(
        env_id="ALE/Breakout-v5",
        total_updates=100_000,
        num_envs=32,
        rollout_len=128,
        gamma=0.99,
        lr=2.5e-4,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=0,
    ):
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.AsyncVectorEnv([make_env(env_id, seed + i) for i in range(num_envs)])
    obs, _ = envs.reset()
    
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(actions=act_dim).to(device)
    model = torch.compile(model) 
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_eval_score = 0

    if RUN_SLATE:
        print("Running Slate client")
        def run_client():
            env = make_env(env_id, 1)()
            agent = SlateAgentPPO(env, 'checkpoints')
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
        act_buf = []

        for t in range(rollout_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
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
            act_buf.append(actions)

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
        actions_flat = torch.stack(act_buf).reshape(T * N)

        advantages = returns - values

        batch_size = T * N
        num_minibatches = 4
        minibatch_size = batch_size // num_minibatches
        epsilon = 0.2

        for ppo_batch in range(4):
            b_inds = torch.randperm(batch_size, device=device)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                md_inds = b_inds[start:end]
                obs_slice = obs_stack[md_inds]

                new_logits, new_values = model(obs_slice)
                new_dist = torch.distributions.Categorical(logits=new_logits)
                new_logp = new_dist.log_prob(actions_flat[md_inds])
                ratio = torch.exp(new_logp - logps[md_inds])

                adv_slice = advantages[md_inds]
                adv_slice = (adv_slice - adv_slice.mean()) / (adv_slice.std() + 1e-8)

                policy_loss = -torch.min(ratio * adv_slice, torch.clip(ratio, 1-epsilon, 1+epsilon) * adv_slice).mean()
                value_loss = 0.5 * (returns[md_inds] - new_values).pow(2).mean()
                entropy_loss = -new_dist.entropy().mean()
                
                loss = policy_loss + vf_coef * value_loss + (ent_coef * entropy_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                if update % 50 == 0 and start == 0 and ppo_batch == 0:
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
            if eval_return >= best_eval_score:
                best_eval_score = eval_return
                print("Current best eval score, saving model...")
                torch.save(model._orig_mod.state_dict(), f'checkpoints/model_{update}.pth')

    envs.close()
    return model


if __name__ == "__main__":
    train()
