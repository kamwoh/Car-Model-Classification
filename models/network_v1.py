import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkV1(nn.Module):
    def __init__(self, base, num_classes):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Linear(in_features, num_classes)
        else: # mobile net v2
            in_features = self.base.last_channel

            self.base.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x):
        fc = self.base(x)
        return fc


class NetworkV2(nn.Module):
    def __init__(self, base, num_classes, num_makes, num_types):
        super().__init__()
        self.base = base

        if hasattr(base, 'fc'):
            in_features = self.base.fc.in_features
            self.base.fc = nn.Sequential()
        else: # mobile net v2
            in_features = self.base.last_channel
            self.base.classifier = nn.Sequential()

        self.brand_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_makes)
        )

        self.type_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_types)
        )

        self.class_fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features + num_makes + num_types, num_classes)
        )

    def forward(self, x):
        out = self.base(x)
        brand_fc = self.brand_fc(out)
        type_fc = self.type_fc(out)

        concat = torch.cat([out, brand_fc, type_fc], dim=1)

        fc = self.class_fc(concat)

        return fc, brand_fc, type_fc
