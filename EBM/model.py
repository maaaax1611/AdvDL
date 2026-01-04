import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ShallowCNN(nn.Module):
    def __init__(self, hidden_features=32, num_classes=42, **kwargs):
        super().__init__()
        c_hid1 = hidden_features
        c_hid2 = hidden_features * 2
        c_hid3 = hidden_features * 4

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            Swish(),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_hid3, num_classes)
        )

    def get_logits(self, x):
        # TODO (3.2): Implement classification procedure that outputs the logits across the classes
        #  Consider using F.adaptive_avg_pool2d to convert between the 2D features and a linear representation.
        
        # extract features using CNN
        z = self.cnn_layers(x)

        # pool spatial dims (H,W) down to (1,1) so it fits the linear layer we use to get logits
        # btw: we use logits for 2 things
        # - energy: log logits <--> high energy
        # - classification
        z =  F.adaptive_avg_pool2d(z, 1)

        # pass through fc-layer to get logits (batch_size, num_classes)
        logits = self.fc_layers(z)

        return logits

    def forward(self, x, y=None) -> torch.Tensor:
        # TODO (3.2): Implement forward function for (1) Unconditional JEM (EBM), (2) Conditional JEM.
        #  (You can also reuse your implementation of 'self.get_logits(x)' if this helps you.)
        logits = self.get_logits(x)

        if y is None:
            # Unconditional JEM / EBM:
            # We marginalize over y using LogSumExp: E(x) = -LogSumExp(f(x))
            # The negative sign is crucial because Energy = -Logit
            energy = -torch.logsumexp(logits, dim=1)
        else:
            # Conditional JEM:
            # We select the logit corresponding to class y: E(x, y) = -f(x)[y]
            # gather helps us pick the specific index per batch element
            # shape of y: (batch_size,) -> (batch_size, 1) for gather
            predicted_logits = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            energy = -predicted_logits
            
        return energy
