from torch import nn


class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, actions=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(8,8), stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            )
        self.policy_head = nn.Linear(3136, actions)
        self.value_head = nn.Linear(3136, 1)


    def forward(self, x):
        z = self.features(x)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)

        return logits, value
    

class Encoder(nn.Module):
    def __init__(self, in_channels=4, actions=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(), # 1152
            )
    

    def forward(self, x):
        return self.features(x)


class ForwardModel(nn.Module):
    def __init__(self, feature_dim=1152, actions=7):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(actions+feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, x):
        return self.decoder(x)


class InverseModel(nn.Module):
    def __init__(self, feature_dim=1152, actions=7):
        super().__init__()
        self.inverse = nn.Sequential(
            nn.Linear(feature_dim*2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, actions),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, x):
        return self.inverse(x)

