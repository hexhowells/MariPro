import gymnasium as gym
import torch


def make_env(env_id: str, seed: int):
    def make():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    
    return make


def render_policy(model, env_id="CartPole-v1", max_steps=1000, seed=123):
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=seed)

    device = next(model.parameters()).device
    total_reward = 0.0

    for _ in range(max_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = model(obs_t)
            action = torch.argmax(logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    env.close()

    return total_reward