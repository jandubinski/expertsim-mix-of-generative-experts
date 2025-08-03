import torch
import torch.nn as nn
from expertsim.models.proton.discriminator import GroupedLinear
from torch.nn import functional as F


class GeneratorUnified(nn.Module):
    def __init__(self, noise_dim, cond_dim, n_experts, image_shape=(56, 30), **kwargs):
        super().__init__()
        self.n_experts = n_experts
        self.image_shape = image_shape
        self.input_dim = noise_dim + cond_dim
        self.output_channels = 1

        # Expert-specific fully connected layers
        self.fc1 = nn.ModuleList([nn.Linear(self.input_dim, 256) for _ in range(n_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(256, 128 * 20 * 10) for _ in range(n_experts)])

        self.upsample = nn.Upsample(scale_factor=(3, 2))


        # self.conv_layers = nn.Sequential(
        #     nn.Conv2d(128 * self.n_experts, 256 * self.n_experts, kernel_size=(2, 2)),
        #     nn.BatchNorm2d(256 * self.n_experts),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Upsample(scale_factor=(1, 2)),
        #     nn.Conv2d(256 * self.n_experts, 128 * self.n_experts, kernel_size=(2, 2)),
        #     nn.BatchNorm2d(128 * self.n_experts),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(128 * self.n_experts, 64 * self.n_experts, kernel_size=(2, 2)),
        #     nn.BatchNorm2d(64 * self.n_experts),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(64 * self.n_experts, 1 * self.n_experts, kernel_size=(2, 7)),
        #     nn.ReLU(inplace=True)
        # )


        self.conv_layers = nn.Sequential(
            nn.Conv2d(128 * n_experts, 64 * n_experts, kernel_size=(3, 4), groups=n_experts),
            nn.BatchNorm2d(64 * n_experts),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(64 * n_experts, 32 * n_experts, kernel_size=(2, 4), groups=n_experts),
            nn.BatchNorm2d(32 * n_experts),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32 * n_experts, 1 * n_experts, kernel_size=(2, 2), groups=n_experts),
            nn.ReLU()
        )

    def forward(self, noise, cond, expert_ids):
        B = noise.size(0)
        x = torch.cat([noise, cond], dim=1)

        # Per-expert processing
        expert_outputs = []
        for i in range(self.n_experts):
            mask = (expert_ids == i)
            if mask.sum() == 0:
                expert_outputs.append(torch.zeros(0, 128, 20, 10, device=noise.device))
                continue

            x_i = x[mask]  # select batch items for this expert
            x_i = F.leaky_relu(self.fc1[i](x_i), 0.1)
            x_i = F.leaky_relu(self.fc2[i](x_i), 0.1)
            x_i = x_i.view(-1, 128, 20, 10)
            expert_outputs.append(x_i)

        # Concatenate all expert outputs back into a [B, 128, H, W] tensor ordered by input
        output = torch.zeros(B, 128, 20, 10, device=noise.device)
        for i, x_i in enumerate(expert_outputs):
            mask = (expert_ids == i)
            if mask.sum() > 0:
                output[mask] = x_i

        # Stack into grouped-conv input format: [B, 128 * n_experts, H, W]
        # Each expert occupies a separate group slice
        grouped_input = torch.zeros(B, 128 * self.n_experts, 20, 10, device=noise.device)
        for i in range(self.n_experts):
            mask = (expert_ids == i)
            if mask.sum() > 0:
                grouped_input[mask, i * 128:(i + 1) * 128, :, :] = output[mask]

        x = self.upsample(grouped_input)  # [B, 128 * n_experts, H', W']
        x = self.conv_layers(x)

        # Reshape to [B, n_experts, 1, H, W]
        # x = x.view(B, self.n_experts, 1, *x.shape[2:])

        # Optional: Mask out inactive experts (if needed for later loss logic)
        # mask = gumbel_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, n_experts, 1, 1, 1]
        # x = x * mask

        return x



### working mix of experts, but there are issues with the groupedlinear layers where the processing is not entirely separate or correct

#
# class GeneratorUnified(nn.Module):
#     def __init__(self, noise_dim, cond_dim, n_experts, image_shape=(56, 30), **kwargs):
#         super().__init__()
#         self.n_experts = n_experts
#         self.image_shape = image_shape
#         self.input_dim = noise_dim + cond_dim
#         self.output_channels = 1
#
#         self.fc1 = GroupedLinear(self.input_dim, 256, n_experts)
#         self.fc2 = GroupedLinear(256, 128 * 20 * 10, n_experts)
#
#         self.upsample = nn.Upsample(scale_factor=(3, 2))
#
#
#         # self.conv_layers = nn.Sequential(
#         #     nn.Conv2d(128 * self.n_experts, 256 * self.n_experts, kernel_size=(2, 2)),
#         #     nn.BatchNorm2d(256 * self.n_experts),
#         #     nn.Dropout(0.2),
#         #     nn.LeakyReLU(0.1, inplace=True),
#         #     nn.Upsample(scale_factor=(1, 2)),
#         #     nn.Conv2d(256 * self.n_experts, 128 * self.n_experts, kernel_size=(2, 2)),
#         #     nn.BatchNorm2d(128 * self.n_experts),
#         #     nn.Dropout(0.2),
#         #     nn.LeakyReLU(0.1, inplace=True),
#         #     nn.Conv2d(128 * self.n_experts, 64 * self.n_experts, kernel_size=(2, 2)),
#         #     nn.BatchNorm2d(64 * self.n_experts),
#         #     nn.Dropout(0.2),
#         #     nn.LeakyReLU(0.1, inplace=True),
#         #     nn.Conv2d(64 * self.n_experts, 1 * self.n_experts, kernel_size=(2, 7)),
#         #     nn.ReLU(inplace=True)
#         # )
#
#
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(128 * n_experts, 64 * n_experts, kernel_size=(3, 4), groups=n_experts),
#             nn.BatchNorm2d(64 * n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Upsample(scale_factor=(1, 2)),
#             nn.Conv2d(64 * n_experts, 32 * n_experts, kernel_size=(2, 4), groups=n_experts),
#             nn.BatchNorm2d(32 * n_experts),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(32 * n_experts, 1 * n_experts, kernel_size=(2, 2), groups=n_experts),
#             nn.ReLU()
#         )
#
#     def forward(self, noise, cond, gumbel_weights):
#         B = noise.size(0)
#         x = torch.cat([noise, cond], dim=2)  # [B, input_dim]
#
#         # Expand across experts: [B, n_experts, input_dim]
#         # x = x.unsqueeze(1).expand(-1, self.n_experts, -1).contiguous()
#
#         x = x.view(B * self.n_experts, -1)
#
#         # === Grouped Linear layers ===
#         x = self.fc1(x)
#         x = F.leaky_relu(x, 0.1)
#         x = self.fc2(x)
#         x = F.leaky_relu(x, 0.1)
#
#         #         x = self.fc2(x)
#         #         x = x.view(-1, 128 * self.n_experts, 20, 10)
#         #         x = self.upsample(x)
#         #         x = self.conv_layers(x)  # Output shape: [batch_size, n_experts, H, W]
#         #
#
#         x = x.view(B, 128 * self.n_experts, 20, 10)  # ← reshape for grouped conv input
#
#         x = self.upsample(x)  # [B, 128 * n_experts, 20, 20]
#         x = self.conv_layers(x)  # apply grouped convs
#
#         # Output: [B, 1 * n_experts, H, W] → reshape to per-expert
#         x = x.view(B, self.n_experts, 1, *x.shape[2:])  # [B, n_experts, 1, H, W]
#
#         # === Apply expert gating mask ===
#         # mask = gumbel_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # [B, n_experts, 1, 1, 1]
#         # # dont add the output of each generator contribution but just strictly keep the output of one generator...
#         # x = x * mask  # zero out unused experts
#
#         return x

# class GeneratorUnified(nn.Module):
#     def __init__(self, noise_dim, cond_dim, di_strength, in_strength, n_experts=3, **kwargs):
#         self.name = "Generator-MultiOutput"
#         self.di_strength = di_strength
#         self.in_strength = in_strength
#         self.n_experts = n_experts
#
#         super(GeneratorUnified, self).__init__()
#         # Keep the exact same architecture as your original
#         self.fc1 = nn.Sequential(
#             nn.Linear(noise_dim + cond_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.Dropout(0.2),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 128 * 20 * 10 * self.n_experts),
#             nn.BatchNorm1d(128 * 20 * 10 * self.n_experts),
#             nn.Dropout(0.2),
#             nn.LeakyReLU(0.1, inplace=True)
#         )
#         self.upsample = nn.Upsample(scale_factor=(3, 2))
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(128 * self.n_experts, 256 * self.n_experts, kernel_size=(2, 2)),
#             nn.BatchNorm2d(256 * self.n_experts),
#             nn.Dropout(0.2),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Upsample(scale_factor=(1, 2)),
#             nn.Conv2d(256 * self.n_experts, 128 * self.n_experts, kernel_size=(2, 2)),
#             nn.BatchNorm2d(128 * self.n_experts),
#             nn.Dropout(0.2),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(128 * self.n_experts, 64 * self.n_experts, kernel_size=(2, 2)),
#             nn.BatchNorm2d(64 * self.n_experts),
#             nn.Dropout(0.2),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(64 * self.n_experts, 1 * self.n_experts, kernel_size=(2, 7)),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, noise, cond):
#         # Same forward pass as original
#         x = torch.cat((noise, cond), dim=1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = x.view(-1, 128 * self.n_experts, 20, 10)
#         x = self.upsample(x)
#         x = self.conv_layers(x)  # Output shape: [batch_size, n_experts, H, W]
#
#         # Reshape to separate the experts into a new dimension
#         # Assuming the output shape is [batch_size, n_experts, H, W]
#         batch_size, n_channels, height, width = x.shape
#
#         # Reshape to [batch_size, n_experts, H, W] to represent separate images
#         # This keeps the exact same tensor, just viewed differently
#         x = x.view(batch_size, self.n_experts, height, width)
#
#         return x


class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, di_strength, in_strength, **kwargs):
        self.name = "Generator-1-original-architecture"
        self.di_strength = di_strength
        self.in_strength = in_strength
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, 256),  # This should be 19 (10 + 9)
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128 * 20 * 10),
            nn.BatchNorm1d(128 * 20 * 10),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.upsample = nn.Upsample(scale_factor=(3, 2))
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(2, 2)),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=(1, 2)),
            nn.Conv2d(256, 128, kernel_size=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 64, kernel_size=(2, 2)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 1, kernel_size=(2, 7)),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise, cond):
        x = torch.cat((noise, cond), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 128, 20, 10)
        x = self.upsample(x)
        x = self.conv_layers(x)
        return x
