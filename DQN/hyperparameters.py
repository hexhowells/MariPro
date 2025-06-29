batch_size = 32
lr = 1e-4

history_len = 4
frame_size = 84
in_channels = 4
action_space = 4
replay_memory_size = 1_000_000

target_update_frequency = 5_000
update_frequency = 4

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay_steps = 500_000 

gamma = 0.99
action_repeats = 1

replay_start_size = 50_000

total_steps = 10_000_000