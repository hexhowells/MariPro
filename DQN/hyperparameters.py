batch_size = 32
lr = 3e-4

history_len = 4
frame_size = 84
replay_memory_size = 10_000

target_update_frequency = 10_000
update_frequency = 4

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
discount_factor = 0.99
action_repeats = 4

replay_start_size = 5000 # random policy is run for this many frames before training

num_episodes = 10_000
episode_len = 20_000
