<p align="center">
  <img src="https://github.com/hexhowells/MariPro/blob/main/banner.png" width=80%>
</p>

Learning to play Super Mario Bros with AI. 

Implements various genetic algorithms and reinforcement learning algorithms to play the game Super Mario Bros using the environment [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros). Platform to mostly learn about different reinforcement learning algorithms using a standardised environment to compare the solutions against each other. Will also train models on Breakout for quicker development or for algorithms that do not work well on the SMB environment.

---

## Current Implemented Methods
- Heuristic Based
- Evolutionary Learning
- Neuroevolution
- [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [DQN](https://www.nature.com/articles/nature14236)
- [DDQN](https://www.nature.com/articles/nature14236)
- Prioritized Experience Replay
- [Dueling DQN](https://arxiv.org/abs/1511.06581)
- [RainbowDQN](https://arxiv.org/abs/1710.02298)
- [A2C](https://arxiv.org/abs/1602.01783)
- [Intrinsic Curiosity Module](https://arxiv.org/abs/1705.05363)
- [Random Network Distillation](https://arxiv.org/abs/1810.12894)

### TODO
- [Proximal policy optimization](https://arxiv.org/abs/1707.06347)
- [MuZero](https://arxiv.org/abs/1911.08265)
- [EfficientZero](https://arxiv.org/abs/2111.00210)
- Deep Deterministic Policy Gradient
- Soft Actor-Critic
- REINFORCE

---

## Notable RAM Addresses
Usefull addresses in the game's RAM for the heuristic method. We use raw pixels and the environment info for all other solutions.

| Addr | Description |
| ---- | ----------- |
| 0x071A | Current Screen |
| 0x071C | Screen offset |
| 0x0500-0x069F | Tilemap (32 x 13 circular buffer) |
| 0x04AC-0x04AF | Player hitbox (x1, y1, x2, y2) |
| 0x04B0-0x04C3 | Enemy hitboxes (x1, y1, x2, y2) x5 |
| 0x000F-0x0013 | Enemy loaded (0=No, 1=Yes) x5 |
