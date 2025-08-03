import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Router Network
class RouterNetwork(nn.Module):
    def __init__(self, cond_dim, n_experts, **kwargs):
        super(RouterNetwork, self).__init__()
        self.name = "router-architecture-1"
        self.n_experts = n_experts
        self.fc_layers = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, self.n_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, cond, tau=1.0, hard=False):
        logits = self.fc_layers(cond)  # [B, E] raw scores
        gates = F.gumbel_softmax(logits, tau=tau, hard=hard)
        # gates now ∈ [0,1]⁽ᴮ⁾ˣᴱ, sums to 1 per batch element;
        # if hard=True, a straight-through one-hot approximation
        return gates, logits

# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim, num_generators):
#         super(RouterNetwork, self).__init__()
#         self.name = "router-architecture-3-gumbel-softmax-megan"
#         # self.description = "Gumbel softmax routing"
#         self.num_generators = num_generators
#         self.fc_layers = nn.Sequential(
#             nn.Linear(cond_dim, 128),
#             nn.LeakyReLU(0.1),
#             nn.Linear(128, 64),
#             nn.LeakyReLU(0.1),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.1),
#             nn.Linear(32, self.num_generators)
#         )
#
#         self.iteration = 0  # Track iterations for temperature annealing
#
#     def compute_temperature(self):
#         """Implements MEGAN's temperature annealing schedule: τ = 0.5 exp(-0.001 × iter)"""
#         return 0.5 * torch.exp(torch.tensor(-0.001 * self.iteration))
#
#     def gumbel_softmax_sample(self, logits, temperature):
#         """Sample from the Gumbel-Softmax distribution"""
#         # Sample from Gumbel(0, 1)
#         gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
#         # Apply gumbel noise
#         y = logits + gumbel_noise
#         # Apply softmax with temperature
#         return F.softmax(y / temperature, dim=-1)
#
#     def straight_through_gumbel_softmax(self, logits, temperature):
#         """Straight-Through Gumbel-Softmax as described in MEGAN paper"""
#         # Regular gumbel softmax for forward pass
#         y_soft = self.gumbel_softmax_sample(logits, temperature)
#
#         # Hard selection for inference
#         index = y_soft.max(-1, keepdim=True)[1]
#         y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
#
#         # Straight-through estimator: use hard selection but backprop through soft
#         return (y_hard - y_soft).detach() + y_soft
#
#     def forward(self, cond, hard=True):
#         self.iteration += 1  # Update iteration counter
#
#         # Get temperature from annealing schedule
#         temperature = self.compute_temperature()
#
#         # Compute logits (assignment scores)
#         logits = self.fc_layers(cond)  # [B, num_generators] raw scores
#
#         # Use the Straight-Through Gumbel-Softmax (always hard for MEGAN)
#         gates = self.straight_through_gumbel_softmax(logits, temperature)
#
#         return gates



class AttentionRouterNetwork(nn.Module):
    def __init__(self, cond_dim, num_experts, num_heads=4, hidden_dim=128):
        """
        cond_dim : dimensionality of the input condition vector.
        num_experts: number of generator/discriminator experts.
        num_heads : number of heads for the multi-head attention.
        hidden_dim : latent dimension to which conditions are projected.
        """
        super(AttentionRouterNetwork, self).__init__()
        self.cond_dim = cond_dim
        self.name = "AttentionRouterNetwork"
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        # Optionally project the condition vector into a hidden representation.
        self.query_proj = nn.Linear(cond_dim, hidden_dim)

        # Self-attention module (using batch_first = True so inputs are (B, S, D))
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Expert keys: a learnable bank of expert embeddings (one per expert)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, hidden_dim))

        # A learnable temperature parameter to adjust the sharpness of the routing distribution.
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, cond):
        # Input: cond of shape (batch_size, cond_dim)
        # Project the condition to get a query representation
        query = self.query_proj(cond)  # shape: (B, hidden_dim)

        # Unsqueeze to add a sequence length (here, we'll use 1, treating each input as a sequence of length 1)
        query_seq = query.unsqueeze(1)  # shape: (B, 1, hidden_dim)

        # Apply self-attention.
        # Since our sequence length is 1, we simply get back a representation of shape (B, 1, hidden_dim)
        attn_output, _ = self.attention(query_seq, query_seq, query_seq)
        attn_output = attn_output.squeeze(1)  # shape: (B, hidden_dim)

        # Compute dot-product scores with each expert's key.
        # This yields a tensor of shape (B, num_experts)
        logits = torch.matmul(attn_output, self.expert_keys.T)

        # Scale the logits by the temperature to adjust their sharpness.
        logits = logits / self.temperature

        # Apply softmax to obtain routing probabilities
        routing_probs = F.softmax(logits, dim=-1)
        return routing_probs


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     """
#     Fixed bottleneck in the last layers. Added batchnor and dropout
#     """
#     def __init__(self, cond_dim, num_generators):
#         super(RouterNetwork, self).__init__()
#         self.name = "router-architecture-2"
#         self.num_generators = num_generators
#         self.fc_layers = nn.Sequential(
#             nn.Linear(cond_dim, 128),
#             nn.BatchNorm1d(128),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(64, 48),  # Smoother transition
#             nn.BatchNorm1d(48),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),  # Dropout
#             nn.Linear(48, 32),
#             nn.BatchNorm1d(32),  # Batch Norm
#             nn.LeakyReLU(0.1),
#             nn.Linear(32, self.num_generators),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim):
#         super(RouterNetwork, self).__init__()
#         self.fc_layers = nn.Sequential(
#             # Hidden Layer 1
#             nn.Linear(cond_dim, 128),
#             nn.BatchNorm1d(128),  # Batch normalization
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),  # Dropout for regularization
#
#             # Hidden Layer 2
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),  # Batch normalization
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.5),  # Dropout for regularization
#
#             # Hidden Layer 3
#             nn.Linear(64, 32),
#             nn.BatchNorm1d(32),  # Batch normalization
#             nn.LeakyReLU(0.1),
#
#             # Output Layer
#             nn.Linear(32, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)


# # Define the Router Network
# class RouterNetwork(nn.Module):
#     def __init__(self, cond_dim):
#         super(RouterNetwork, self).__init__()
#         self.fc_layers = nn.Sequential(
#             # Hidden Layer 1
#             nn.Linear(cond_dim, 32),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),
#
#             # Output Layer
#             nn.Linear(32, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, cond):
#         return self.fc_layers(cond)