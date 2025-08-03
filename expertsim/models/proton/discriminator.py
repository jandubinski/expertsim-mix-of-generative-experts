import torch
import torch.nn as nn
import math
from torch.nn import functional as F

#
# class DiscriminatorUnified(nn.Module):
#     def __init__(self, cond_dim, n_experts, **kwargs):
#         super(DiscriminatorUnified, self).__init__()
#         self.name = "Discriminator-3-unified"
#         self.n_experts = n_experts
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1*self.n_experts, 32*self.n_experts, kernel_size=(3, 3)),
#             nn.BatchNorm2d(32*self.n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 2)),
#             nn.Conv2d(32*self.n_experts, 16*self.n_experts, kernel_size=(3, 3)),
#             nn.BatchNorm2d(16*self.n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2),
#             nn.MaxPool2d(kernel_size=(2, 1))
#         )
#         self.fc1 = nn.Sequential(
#             nn.Linear(16 * 12 * 12*self.n_experts + cond_dim, 128*self.n_experts),
#             nn.BatchNorm1d(128*self.n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(128*self.n_experts, 64*self.n_experts),
#             nn.BatchNorm1d(64*self.n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Dropout(0.2)
#         )
#         self.fc3 = nn.Linear(64*self.n_experts, 1*self.n_experts)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, img, cond):
#         x = self.conv_layers(img)
#         # print("conv layers", x.shape)  # Debugging line to check the shape after conv layers
#         x = x.view(x.size(0), -1)
#         # print("conv view", x.shape)  # Debugging line to check the shape after conv layers
#         x = torch.cat((x, cond), dim=1)
#         # print("cat shape", x.shape)  # Debugging line to check the shape after conv layers
#         x = self.fc1(x)
#         # print("f1 shape", x.shape)  # Debugging line to check the shape after conv layers
#         latent = self.fc2(x)
#         out = self.fc3(latent)
#         latent = latent.view(-1, self.n_experts, 64)  # Reshape to [batch_size, n_experts]
#         out = self.sigmoid(out)
#         return out, latent


class GroupedLinear(nn.Module):
    def __init__(self, in_features, out_features, n_experts, mode='generator'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_experts = n_experts
        self.weight = nn.Parameter(torch.Tensor(n_experts, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(n_experts, out_features))
        self.reset_parameters()
        self.mode = mode

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if self.mode != 'generator':
            # Create fresh parameter copies for each forward pass to avoid version conflicts
            weight_copy = self.weight.clone()
            bias_copy = self.bias.clone()

            result = torch.einsum('bgi,goi->bgo', x, weight_copy)
            bias_expanded = bias_copy.unsqueeze(0).expand(result.shape[0], -1, -1).clone()
            return result + bias_expanded
        else:
            B_total, in_dim = x.size()
            x = x.view(self.n_experts, -1, in_dim)  # [n_experts, B, in_features]

            # Also create copies for generator mode for consistency
            weight_copy = self.weight.clone()
            bias_copy = self.bias.clone()

            out = torch.einsum("ebi,eoi->ebo", x, weight_copy) + bias_copy.unsqueeze(1)
            return out.view(B_total, self.out_features)


class DiscriminatorUnified(nn.Module):
    def __init__(self, cond_dim, n_experts=3, **kwargs):
        super().__init__()
        self.n_experts = n_experts

        self.conv1 = nn.Conv2d(n_experts, 32 * n_experts, kernel_size=3, groups=n_experts)
        self.gn1 = nn.GroupNorm(num_groups=n_experts, num_channels=32 * n_experts)
        self.conv2 = nn.Conv2d(32 * n_experts, 16 * n_experts, kernel_size=3, groups=n_experts)
        self.gn2 = nn.GroupNorm(num_groups=n_experts, num_channels=16 * n_experts)

        self.dropout = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d((2, 1))

        self.cond_fc = GroupedLinear(cond_dim, 64, n_experts, mode='discriminator')
        self.fc1 = GroupedLinear(16 * 12 * 12 + 64, 128, n_experts, mode='discriminator')
        self.fc2 = GroupedLinear(128, 64, n_experts, mode='discriminator')
        self.fc3 = GroupedLinear(64, 1, n_experts, mode='discriminator')

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, cond, gumbel_weights=None):
        """
        x: [B, n_experts, 1, H, W] (each expert gets its own image)
        cond: [B, cond_dim]
        gumbel_weights: [B, n_experts] (only for routing — not used inside forward)
        """

        B, E, C, H, W = x.shape
        assert E == self.n_experts, "Mismatch in expert count"

        # Merge expert dim into channel dim → [B, E*C, H, W]
        x = x.view(B, E * C, H, W)

        x = self.conv1(x)
        x = self.gn1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)
        x = self.pool2(x)

        # Reshape to [B, E, -1]
        x = x.view(B, self.n_experts, -1)  # [B, E, 16*12*12]

        # Process condition for each expert
        cond_exp = cond.unsqueeze(1).expand(-1, self.n_experts, -1)  # [B, E, cond_dim]
        cond_embed = self.cond_fc(cond_exp)  # [B, E, 64]

        # Concatenate features + cond
        x = torch.cat([x, cond_embed], dim=2)  # [B, E, features + cond]

        x = self.fc1(x)  # [B, E, 128]
        x = F.leaky_relu(x, 0.1)
        x = self.dropout(x)

        latent = self.fc2(x)  # [B, E, 64]
        x = F.leaky_relu(latent, 0.1)
        x = self.dropout(x)

        out = self.fc3(x)  # [B, E, 1]
        out = self.sigmoid(out)

        return out, latent
#
# class DiscriminatorUnified(nn.Module):
#     def __init__(self, n_experts=3, cond_dim=10, input_channels=1, **kwargs):
#         super().__init__()
#         self.n_experts = n_experts
#         self.input_channels = input_channels
#
#         self.cond_fc = GroupedLinear(cond_dim, 128, n_experts, mode='discriminator')
#
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d((input_channels + 128) * n_experts, 64 * n_experts,
#                       kernel_size=3, padding=1, groups=n_experts),
#             nn.BatchNorm2d(64 * n_experts),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(64 * n_experts, 32 * n_experts,
#                       kernel_size=3, padding=1, groups=n_experts),
#             nn.BatchNorm2d(32 * n_experts),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool
#         self.final_fc = GroupedLinear(32, 1, n_experts, mode='discriminator')
#
#     def forward(self, x, cond, gumbel_weights):
#         """
#         x: Tensor of shape [B, n_experts, C, H, W]
#         cond: Tensor of shape [B, cond_dim]
#         gumbel_weights: Tensor of shape [B, n_experts]
#         """
#         B, n_experts, C, H, W = x.shape
#         assert n_experts == self.n_experts, "Mismatch in number of experts"
#
#         # Reshape input: [B, n_experts, C, H, W] → [B, C*n_experts, H, W]
#         x = x.view(B, C * n_experts, H, W)
#
#         # Process condition: → [B, n_experts, 128]
#         cond = self.cond_fc(cond.unsqueeze(1).expand(-1, n_experts, -1))  # [B, n_experts, cond_dim] → [B, n_experts, 128]
#
#         # Prepare condition for concatenation with image
#         cond = cond.permute(0, 2, 1).contiguous().view(B, 128 * n_experts, 1, 1)  # [B, 128*n_experts, 1, 1]
#         cond = cond.expand(-1, -1, H, W)  # [B, 128*n_experts, H, W]
#
#         # Concatenate image and condition
#         x = torch.cat([x, cond], dim=1)  # [B, (C+128)*n_experts, H, W]
#
#         # Pass through conv layers
#         x = self.conv_layers(x)  # [B, 32*n_experts, H, W]
#         x = self.pool(x).view(B, n_experts, 32)  # [B, n_experts, 32]
#
#         # Final grouped linear layer
#         logits = self.final_fc(x).squeeze(-1)  # [B, n_experts]
#
#         # MoE weighted sum
#         out = (logits * gumbel_weights).sum(dim=1, keepdim=True)  # [B, 1]
#         return out


class Discriminator(nn.Module):
    def __init__(self, cond_dim, **kwargs):
        super(Discriminator, self).__init__()
        self.name = "Discriminator-4-removed-sigmoid-bce-with-logits"
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=(2, 1))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 12 * 12 + cond_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(64, 1)

    def forward(self, img, cond):
        x = self.conv_layers(img)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond), dim=1)
        x = self.fc1(x)
        latent = self.fc2(x)
        out = self.fc3(latent)
        return out, latent