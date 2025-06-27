# gamma = 0.99
# epsilon = 1.0
# epsilon_min = 0.01
# epsilon_decay = 0.995
# batch_size = 64
# replay_memory_size = 10_000
# target_update_frequency = 5000
# lr = 1e-5
# num_episodes = 1000
# action_repeats = 4
# buffer_size = 50
# frame_size = 84
# env_stop_threshold = 2000


batch_size = 32
lr = 0.00025

history_len = 4
frame_size = 84
replay_memory_size = 1_000_000

target_update_frequency = 10_000
update_frequency = 4

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
discount_factor = 0.99
action_repeats = 4

reply_start_size = 50_000 # random policy is run for this many frames before training

num_episodes = 100_000
episode_len = 20_000
