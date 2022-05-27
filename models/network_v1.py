import torch.nn as nn


class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes)
        else:  # mobile net v2
            in_features = self.base.last_channel

            self.base.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x):
        fc = self.base(x)
        return fc
