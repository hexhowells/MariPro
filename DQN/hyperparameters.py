batch_size = 64
lr = 3e-4

history_len = 4
frame_size = 84
in_channels = 4
action_space = 7
replay_memory_size = 100_000

target_update_frequency = 5_000
update_frequency = 2

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
discount_factor = 0.99
action_repeats = 4

reply_start_size = 50_000 # random policy is run for this many frames before training

num_episodes = 100_000
episode_len = 20_000
