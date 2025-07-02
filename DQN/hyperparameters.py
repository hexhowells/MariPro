# Environment & Input
frame_size = 84
history_len = 4
in_channels = 4
action_space = 4

# Training Parameters
batch_size = 32
lr = 1e-4  # 0.00001
gamma = 0.99
update_frequency = 4
target_update_frequency = 10_000
replay_start_size = 50_000
replay_memory_size = 100_000

# Exploration (Epsilon-Greedy)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay_steps = 1_000_000

# Training Schedule
total_steps = 50_000_000
eval_steps = 50_000
checkpoint_steps = 100_000