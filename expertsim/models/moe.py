import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations
from expertsim.train.utils import calculate_expert_utilization_entropy, calculate_expert_distribution_loss, calculate_adaptive_load_balancing_loss
from expertsim.train.utils import sum_channels_parallel, calculate_joint_ws_across_experts, generate_and_save_images
import pandas as pd
from time import time
import wandb

# import deepspeed
# from deepspeed.moe.layer import MoE
#
#
# class MoEWrapperUnified(nn.Module):
#     def __init__(self, generator, discriminator, router, n_experts):
#         super().__init__()
#         self.generator_experts = nn.ModuleList([
#             generator
#             for _ in range(n_experts)
#         ])
#         self.moe_layer = MoE(
#             gate=router,
#             experts=self.generator_experts,
#             loss_coef=1e-2,
#             top_k=1
#         )
#
#     def forward(self, noise, noise_2, cond):
#         # Combine inputs per sample into one tensor
#         x = torch.cat([noise, cond], dim=-1)
#         out, gate_loss = self.moe_layer(x)
#         return out, gate_loss


class MoEWrapperUnified(nn.Module):
    name = ""
    def __init__(self, generator, discriminator, router, n_experts):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.router = router
        self.n_experts = n_experts

    def forward(self, z_1, z_2, cond):
        # Routing: get Gumbel-Softmax weights
        gumbel_softmax, _ = self.router(cond.clone())
        _, expert_ids = gumbel_softmax.max(dim=1)

        # Build binary mask from expert_assignments
        # Shape: [B, E], one-hot vector where 1 = selected expert
        # batch_size = expert_ids.size(0)
        # mask = F.one_hot(expert_ids, num_classes=self.n_experts).float()  # [B, E]

        # Optionally detach if you don’t want gradients to pass through router for the mask
        # mask = mask.detach()

        # Generate images
        # detach to prevent gradient flowing back to the router
        # z_1_masked = z_1.unsqueeze(1).expand(-1, self.n_experts, -1) * mask.unsqueeze(2)
        # z_2_masked = z_2.unsqueeze(1).expand(-1, self.n_experts, -1) * mask.unsqueeze(2)
        # cond_masked = cond.unsqueeze(1).expand(-1, self.n_experts, -1) * mask.unsqueeze(2)

        generated_1 = self.generator(z_1, cond, expert_ids.clone().detach())
        generated_2 = self.generator(z_2, cond, expert_ids.clone().detach())

        # mask_expanded = mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Expand mask to [B, E, 1, 1] for broadcasting

        # Apply the same mask to both outputs
        # masked_generated_1 = generated_1 * mask_expanded
        # masked_generated_2 = generated_2 * mask_expanded

        """
        masked_generated_1[0,:,:,:,:].sum(dim=(2, 3))
        Out[2]:
        tensor([[ 0.0000],
                [ 0.0000],
                [50.0121]], device='cuda:0', grad_fn=<SumBackward1>)
        """

        return {
            'generated_1': generated_1,
            'generated_2': generated_1,
            'gates': gumbel_softmax,
            'expert_ids': expert_ids,
        }

    # def forward(self, z_1, z_2, cond, real_images):
    #     # Routing: get Gumbel-Softmax weights
    #     gumbel_softmax, _ = self.router(cond.clone())
    #     _, expert_assignments = gumbel_softmax.max(dim=1)
    #
    #     # Generate images
    #     generated_1 = self.generator(z_1, cond.clone(), gumbel_softmax.clone().detach())
    #     generated_2 = self.generator(z_2, cond.clone(), gumbel_softmax.clone().detach())
    #
    #     # Expand real images to [B, n_experts, 1, H, W]
    #     real_images_exp = real_images.unsqueeze(1).expand(-1, self.n_experts, -1, -1, -1).contiguous()
    #
    #     # Discriminator on real and fake
    #     disc_fake_out, disc_fake_latent = self.discriminator(generated_1, cond.clone(), gumbel_softmax.clone().detach())
    #     disc_fake_out_2, disc_fake_latent_2 = self.discriminator(generated_2, cond.clone(), gumbel_softmax.clone().detach())
    #     disc_real_out, disc_real_latent = self.discriminator(real_images_exp, cond.clone(), gumbel_softmax.clone().detach())
    #
    #     return {
    #         'generated_1': generated_1,
    #         'generated_2': generated_2,
    #         'disc_real_out': disc_real_out,
    #         'disc_real_latent': disc_real_latent,
    #         'disc_fake_out': disc_fake_out,
    #         'disc_fake_out_2': disc_fake_out_2,
    #         'disc_fake_latent': disc_fake_latent,
    #         'disc_fake_latent_2': disc_fake_latent_2,
    #         'gates': gumbel_softmax,
    #         'expert_assignments': expert_assignments,
    #     }
    #

# class MoEWrapperUnified(nn.Module):
#     """
#     Updated Mixture of Experts wrapper using single sparse generator and discriminator.
#     The generator outputs multi-channel images (one channel per expert).
#     Sparsity is enforced by masking non-assigned expert channels.![](../../../../../AppData/Local/Temp/aq05bDV0_700w_0.jpg)
#     """
#
#     def __init__(self, generator_cls, discriminator_cls, aux_reg_cls, router_cls, n_experts: int,
#                  image_shape: tuple = (56, 30)):
#         super().__init__()
#         self.image_shape = image_shape
#         self.generator = generator_cls  # Single generator with multi-expert output
#         self.discriminator = discriminator_cls  # Single discriminator
#         self.router = router_cls
#         self.aux_reg = aux_reg_cls
#         self.n_experts = n_experts
#
#     @staticmethod
#     def apply_expert_mask_vectorized(multi_expert_images, gumbel_weights, latent=False):
#         """Vectorized expert masking without loops"""
#         # Reshape for broadcasting [B, E, 1, 1]
#         # print("multi_expert_images shape", multi_expert_images.shape)
#
#         one_hot = torch.zeros_like(gumbel_weights.clone().detach())  # Detach so the gradients don't flow to the router. Shape: (128, 3)
#         _, expert_assignments = torch.max(gumbel_weights, 1)  # Hard assignments for reference
#         one_hot[torch.arange(gumbel_weights.shape[0]), expert_assignments] = 1.0  # Set 1 at max positions
#         if not latent:
#             mask = one_hot.unsqueeze(2).unsqueeze(3)  # Shape: (128, 3, 1, 1)
#         else:
#             mask = one_hot.unsqueeze(2)  # Shape: (128, 3, 1, 1)
#         # IF hard is false and we want to combine the outputs
#         # weights = gumbel_weights.view(*gumbel_weights.shape, 1, 1)
#
#         weighted_output = multi_expert_images * mask  # torch.Size([B, n_experts, 56, 30])
#         return weighted_output
#
#     def forward(self, z_1: torch.Tensor, z_2: torch.Tensor, cond: torch.Tensor, real_images: torch.Tensor,
#                 device: str = 'cuda'):
#         """
#         Comprehensive forward pass computing all required outputs for training.
#
#         Args:
#             z_1: Primary noise tensor (batch_size, noise_dim) for first generation.
#             z_2: Secondary noise tensor (batch_size, noise_dim) for second generation.
#             cond: Condition tensor (batch_size, cond_dim).
#             real_images: Real images tensor (batch_size, 1, 56, 30) for discrimination.
#
#         Returns:
#             Dictionary with:
#                 - 'generated_1': Tensor (batch_size, n_experts, 56, 30) - sparse, multi-channel output
#                 - 'generated_2': Tensor (batch_size, n_experts, 56, 30)
#                 - 'disc_real_outputs': Tensor (batch_size, n_experts) - per-expert logits for real
#                 - 'disc_fake_outputs': Tensor (batch_size, n_experts) - per-expert logits for fake
#                 - 'gates': Gating scores (batch_size, n_experts)
#                 - 'expert_assignments': Assignments (batch_size,)
#                 - 'aux_reg_outputs': Tensor (batch_size, n_experts, aux_dim) - per-expert aux outputs
#         """
#         batch_size = cond.shape[0]
#         device = cond.device
#
#         # Get gating from router (soft probabilities for sparsity)
#         # gates = self.router(cond)  # Shape: (batch_size, n_experts)
#         # _, expert_assignments = torch.max(gates, 1)  # Hard assignments for reference
#         # _, expert_assignments = torch.max(gates, 1)  # Hard assignments for reference
#
#         gumbel_softmax, router_logits = self.router(cond)
#         # predicted_expert = gumbel_softmax.argmax(dim=1)  # (B, 1)
#         _, expert_assignments = torch.max(gumbel_softmax, 1)  # Hard assignments for reference
#
#         outputs = {
#             'gates': gumbel_softmax,
#             'expert_assignments': expert_assignments,
#             'generated_1': None,
#             'generated_2': None,
#             'disc_real_outputs': None,
#             'disc_fake_outputs': None,
#             'aux_reg_outputs': None
#         }
#         # Create one-hot tensor
#         # one_hot = torch.zeros_like(gates.clone().detach())  # Detach so the gradients don't flow to the router. Shape: (128, 3)
#         # one_hot[torch.arange(gates.shape[0]), expert_assignments] = 1.0  # Set 1 at max positions
#         # mask = one_hot.unsqueeze(2).unsqueeze(3)  # Shape: (128, 3, 1, 1)
#
#         # Generation for primary noise (single call to sparse generator)
#         # generated_1_raw_temp = generated_1_raw.clone() * mask
#         generated_1_raw = self.generator(z_1, cond)  # Shape: (batch_size, n_experts, 56, 30)
#         generated_1_raw_temp = self.apply_expert_mask_vectorized(generated_1_raw, gumbel_softmax)
#         outputs['generated_1'] = generated_1_raw_temp
#
#         # Generation for secondary noise
#         # generated_2_raw_temp = generated_2_raw.clone() * mask
#         generated_2_raw = self.generator(z_2, cond)
#         generated_2_raw = self.apply_expert_mask_vectorized(generated_2_raw, gumbel_softmax)
#         outputs['generated_2'] = generated_2_raw
#
#         # # Discrimination on real images (single call to sparse discriminator)
#         disc_input_real = real_images.expand(-1, self.n_experts, -1, -1)
#         disc_input_real = self.apply_expert_mask_vectorized(disc_input_real, gumbel_softmax)
#         disc_out_real, disc_real_latent = self.discriminator(disc_input_real, cond)  # Shape: (batch_size, n_experts)
#         outputs['disc_real_outputs'] = disc_out_real  # Apply gating for sparsity
#         outputs['disc_real_latents'] = disc_real_latent  # Apply gating for sparsity
#
#         # Discrimination on fake images (using primary generated)
#
#         disc_fake_outputs, disc_fake_latent_1 = self.discriminator(generated_1_raw, cond)
#         disc_fake_latent_1 = self.apply_expert_mask_vectorized(disc_fake_latent_1, gumbel_softmax, latent=True)
#         disc_fake_outputs_temp = disc_fake_outputs.clone()
#         # disc_fake_outputs (128, 3)
#         # disc_fake_latent_1 (128, 64)
#         # gates (128, 3)
#         #
#         outputs['disc_fake_outputs'] = disc_fake_outputs_temp  # Apply gating for sparsity
#         outputs['disc_fake_latents'] = disc_fake_latent_1  # Apply gating for sparsity
#
#         _, disc_fake_latent_2 = self.discriminator(generated_2_raw, cond)
#         disc_fake_latent_2 = self.apply_expert_mask_vectorized(disc_fake_latent_2, gumbel_softmax, latent=True)
#         outputs['disc_fake_latents_2'] = disc_fake_latent_2  # Apply gating for sparsity
#
#         # outputs['disc_real_outputs'] = disc_real_outputs
#         # outputs['disc_fake_outputs'] = disc_fake_outputs
#         # outputs['disc_fake_latents'] = disc_fake_latents
#         # outputs['disc_fake_latents_2'] = disc_fake_latents_2
#
#
#         # Auxiliary regressor on primary generated (per-channel)
#         aux_input = generated_1_raw.view(batch_size * self.n_experts, 1,
#                                          *self.image_shape)  # Flatten for batch processing
#         aux_raw = self.aux_reg(aux_input)
#         outputs['aux_reg_outputs'] = aux_raw.view(batch_size, self.n_experts, -1)  # Reshape back
#         outputs['aux_reg_outputs_features'] = self.aux_reg.get_features(aux_input)
#
#         return outputs
#
#     def get_expert_assignment_counts(self, expert_assignments: torch.Tensor) -> torch.Tensor:
#         """Get count of samples assigned to each expert."""
#         class_counts = torch.zeros(self.n_experts, dtype=torch.float, device=expert_assignments.device)
#         for expert_idx in range(self.n_experts):
#             class_counts[expert_idx] = (expert_assignments == expert_idx).sum().item()
#         return class_counts / expert_assignments.size(0)


class MoEWrapper(nn.Module):
    """
    Mixture of Experts wrapper that encapsulates the routing and expert logic.
    Keeps models separate from optimizers for clean architecture.
    """
    name = "separate-gen-disc-shared-aux-reg"
    description = "MoEWrapper where expert is defined as a generator and discriminator." \
                  "Auxiliary regressor "

    def __init__(self, generator_cls, discriminator_cls, aux_reg_cls, router_cls, n_experts: int, cfg,
                 image_shape: tuple = (56, 30)):
        super().__init__()
        self.image_shape = image_shape
        self.generators = nn.ModuleList([generator_cls for _ in range(n_experts)])
        self.discriminators = nn.ModuleList([discriminator_cls for _ in range(n_experts)])
        self.router = router_cls
        self.aux_reg = aux_reg_cls
        self.n_experts = n_experts
        self.criterion = nn.BCEWithLogitsLoss()
        self.noise_dim = cfg.model.noise_dim
        self.cfg = cfg

    def train_step(self, epoch, cond, real_images, true_positions, std, intensity, aux_reg_optimizer, generator_optimizers,
                   discriminator_optimizers, router_optimizer, device):

        B = cond.size(0)
        # Train router network
        self.router.zero_grad()
        # Get predicted experts assignments for samples. Outputs are the probabilities of each expert for each sample. Shape: (batch, self.n_experts)
        gates, logits = self.router(cond)
        _, predicted_expert = torch.max(gates, 1)  # (B, 1)

        # calculate the class counts for each expert
        class_counts = torch.zeros(self.n_experts, dtype=torch.float).to(device)
        for class_label in range(self.n_experts):
            class_counts[class_label] = (predicted_expert == class_label).sum().item()
        class_counts_adjusted = class_counts / predicted_expert.size(0)

        # train experts
        gen_losses = torch.zeros(self.n_experts, requires_grad=True).to(device)  # hold tensors that will be used for router loss calculation
        disc_losses = torch.zeros(self.n_experts, requires_grad=True).to(device)
        div_losses = np.zeros(self.n_experts)
        aux_reg_losses = np.zeros(self.n_experts)
        intensity_losses = np.zeros(self.n_experts)
        mean_intensities_experts = np.zeros(self.n_experts)  # mean intensities for each expert for each batch
        std_intensities_experts = np.zeros(self.n_experts)  # std intensities for each expert for each batch
        mean_intensities_in_batch_expert = torch.zeros(B, device=device)  # this should contain leaf tensors, as we don't want routers from gradient to propagate to generators

        aux_reg_features_experts = []
        for i in range(self.n_experts):
            selected_indices = (predicted_expert == i).nonzero(as_tuple=True)[0]
            B = len(selected_indices)
            # Generator predictions that will be used to train discriminator and generator
            selected_generator = self.generators[i]
            selected_cond = cond[selected_indices]
            noise_1 = torch.randn(B, self.noise_dim, device=real_images.device)
            fake_images = selected_generator(noise_1, selected_cond)
            #
            # Train discriminator
            #
            if selected_indices.numel() <= 1:
                disc_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                gen_losses[i] = torch.tensor(0.0, requires_grad=True).to(device)
                feature_shape_aux_conv_channels = 64
                aux_reg_features_experts.append(
                    torch.zeros((1, feature_shape_aux_conv_channels), requires_grad=True).to(device))
                continue

            # Clone or detach tensors to avoid in-place modifications
            selected_discriminator = self.discriminators[i]
            selected_discriminator_optimizer = discriminator_optimizers[i]
            selected_real_images = real_images[selected_indices]
            selected_class_counts = class_counts_adjusted[i]

            disc_loss = self.discriminator_train_step(selected_discriminator, fake_images,
                                                      selected_discriminator_optimizer,
                                                      self.criterion,
                                                      selected_class_counts,
                                                      selected_real_images,
                                                      selected_cond, B)
            disc_losses[i] = disc_loss

            #
            # Train each generator
            #
            selected_cond = cond[selected_indices]
            selected_true_positions = true_positions[selected_indices]
            selected_intensity = intensity[selected_indices]
            selected_std = std[selected_indices]
            selected_generator = self.generators[i]
            selected_generator_optimizer = generator_optimizers[i]
            selected_discriminator = self.discriminators[i]
            selected_aux_reg = self.aux_reg
            selected_aux_reg_optimizer = aux_reg_optimizer
            selected_class_counts = class_counts_adjusted[i]

            gen_loss, div_loss, intensity_loss, \
            aux_reg_loss, std_intensity, mean_intensity, mean_intensities, aux_reg_features = self.generator_train_step(
                fake_images,
                noise_1,
                selected_generator,
                selected_discriminator,
                selected_aux_reg,
                selected_cond,
                selected_generator_optimizer,
                selected_aux_reg_optimizer,
                self.criterion,
                selected_true_positions,
                selected_std,
                selected_intensity,
                selected_class_counts,
                len(selected_indices))

            aux_reg_features_experts.append(aux_reg_features)
            mean_intensities_in_batch_expert[
                selected_indices] = mean_intensities.clone().detach().squeeze()  # input the mean intensities for calculated samples

            # Save statistics
            mean_intensities_experts[i] = mean_intensity
            std_intensities_experts[i] = std_intensity
            gen_losses[i] = gen_loss
            div_losses[i] = div_loss
            intensity_losses[i] = intensity_loss
            aux_reg_losses[i] = aux_reg_loss

        #
        # Calculate router loss
        #
        if self.n_experts > 1:
            gan_loss_scaled = (gen_losses.mean() + disc_losses.mean()) * self.cfg.model.router.gan_strength
            expert_entropy_loss = -1 * calculate_expert_utilization_entropy(gates.clone(),
                                                                       self.cfg.model.router.util_strength) if self.cfg.model.router.util_strength != 0 else torch.tensor(
                0.0,
                requires_grad=False,
                device=gates.device)

            expert_distribution_loss = calculate_expert_distribution_loss(gates.clone(),
                                                                          mean_intensities_in_batch_expert.reshape(-1,
                                                                                                                   1),
                                                                          self.cfg.model.router.ed_strength) if self.cfg.model.router.ed_strength != 0. else torch.tensor(
                0.0, requires_grad=False)

            def compute_cross_generator_similarity(reps_i, reps_j):
                """
                Compute similarity between two generators' representations
                reps_i: (batch_i, 64), reps_j: (batch_j, 64)
                """
                # Normalize representations
                reps_i_norm = F.normalize(reps_i, dim=1)  # (batch_i, 64)
                reps_j_norm = F.normalize(reps_j, dim=1)  # (batch_j, 64)

                # Compute cosine similarity matrix
                similarity_matrix = torch.mm(reps_i_norm, reps_j_norm.t())  # (batch_i, batch_j)

                # Option 1: Mean of all pairwise similarities
                # mean_similarity = similarity_matrix.mean()

                # Option 2: Maximum similarity (encourages generators to be different)
                max_similarity = similarity_matrix.max()

                # Option 3: Top-k mean (focus on most similar pairs)
                # k = min(similarity_matrix.numel(), 10)
                # top_k_similarities = torch.topk(similarity_matrix.flatten(), k).values
                # top_k_mean = top_k_similarities.mean()

                return max_similarity  # or mean_similarity or top_k_mean

            def compute_differentiation_loss(aux_reg_features_experts):
                """
                representations_dict: {generator_id: tensor of shape (batch_size_i, 64)}
                Returns diversity loss encouraging different outputs between generators
                """
                total_loss = 0.0
                generator_pairs = 0
                num_experts = len(aux_reg_features_experts)

                for i, j in combinations(range(num_experts), 2):
                    gen_i_reps = aux_reg_features_experts[i]
                    gen_j_reps = aux_reg_features_experts[j]

                    # Compute inter-generator similarity
                    inter_gen_similarity = compute_cross_generator_similarity(
                        gen_i_reps, gen_j_reps
                    )

                    # Minimize inter-generator similarity
                    total_loss += inter_gen_similarity
                    generator_pairs += 1

                return total_loss / generator_pairs if generator_pairs > 0 else 0.0

            # DIFFERENTIATION LOSS
            # def compute_differentiation_loss(aux_reg_features_experts):
            #     """
            #     Compute differentiation loss for all experts based on the feature vectors from convolution layers.
            #     param: discriminator_features: List of tensors containing the features of each expert
            #     return: Differentiation loss
            #     """
            #     loss = torch.zeros(1, device=device)
            #     num_experts = len(aux_reg_features_experts)
            #     # print('+++++++++')
            #     # print("Feature means Experts", aux_reg_features_experts[0].shape)
            #     # # with torch.no_grad():  # Detach computations from graph to save memory
            #     feature_means = [feat.mean(0, keepdim=True) for feat in aux_reg_features_experts]
            #     # # feature_vars = [feat.var(dim=0) for feat in aux_reg_features_experts]
            #     # print('Feature means shape:', [feat.shape for feat in feature_means])
            #     # print('Feature means:', feature_means)
            #     # test to optimize experts, not router
            #
            #     # print("shape means", feature_means[0].shape)
            #     # print("shape vars", feature_vars[0].shape)
            #
            #     for i, j in combinations(range(num_experts), 2):
            #         # Reattach to computation graph only for final loss calculation
            #         mean_i = feature_means[i]
            #         mean_j = feature_means[j]
            #
            #         # var_i = feature_vars[i].detach().requires_grad_(True)
            #         # var_j = feature_vars[j].detach().requires_grad_(True)
            #         # cos_diss_means = 1- F.cosine_similarity(mean_i, mean_j)
            #         # cos_diss_vars = 1- F.cosine_similarity(mean_i, mean_j)
            #         # dissimilarity = 0.8*cos_diss_means + 0.2*cos_diss_vars
            #
            #         cosine_similarity = F.cosine_similarity(mean_i, mean_j)
            #         # -1: vectors dissimilar
            #         # 0: vectors orthogonal
            #         # 1: vectors exactly the same
            #
            #         # thus we want to get minimize the sum of cosine similarities
            #
            #         # dissimilarity = torch.abs(cosine_similarity)
            #
            #         loss -= cosine_similarity
            #     return loss

            differentiation_loss = compute_differentiation_loss(
                aux_reg_features_experts) if self.cfg.model.router.diff_strength != 0. else torch.tensor(0.0)
            differentiation_loss = differentiation_loss * self.cfg.model.router.diff_strength
            # OLD BASED ON MEAN PHOTONSUMS
            # Compute differentiation loss for all experts
            # differentiation_loss_intensities = sum(
            #     np.abs((mean_intensities_experts[i] - mean_intensities_experts[j]))
            #     for i, j in combinations(range(self.n_experts), 2)  # Generate all unique pairs of experts
            # ) if DIFF_STRENGTH != 0. else torch.tensor(0.0)
            # differentiation_loss_stds = sum(
            #     np.abs((std_intensities_experts[i] - std_intensities_experts[j]))
            #     for i, j in combinations(range(self.n_experts), 2)  # Generate all unique pairs of experts
            # ) if DIFF_STRENGTH != 0. else torch.tensor(0.0)
            #
            # differentiation_loss = (differentiation_loss_intensities+differentiation_loss_stds) * DIFF_STRENGTH

            routing_scores = gates.sum(dim=0)
            alb_loss = calculate_adaptive_load_balancing_loss(routing_scores,
                                                              self.cfg.model.router.alb_strength) if self.cfg.model.router.alb_strength != 0. else torch.tensor(0.0)

            # na pewno musi byc detach
            router_loss = gan_loss_scaled + expert_distribution_loss + expert_entropy_loss + alb_loss + differentiation_loss
            if self.cfg.model.router.stop_router_training_epoch is not None:
                if epoch < self.cfg.model.router.stop_router_training_epoch:
                    # Train Router Network
                    router_loss.backward()
                    router_optimizer.step()
                else:
                    router_loss = torch.tensor(0.0)
        else:
            gan_loss_scaled = torch.tensor(0.0)
            router_loss = torch.tensor(0.0)
            expert_distribution_loss = torch.tensor(0.0)
            differentiation_loss = torch.tensor(0.0)
            expert_entropy_loss = torch.tensor(0.0)
            alb_loss = torch.tensor(0.0)

        gen_losses = [gen_loss.item() for gen_loss in gen_losses]
        disc_losses = [disc_loss.item() for disc_loss in disc_losses]
        class_counts = [class_count.item() for class_count in class_counts]
        div_losses = [div_loss.item() for div_loss in div_losses]

        log_metrics = {
        'gen_loss': np.mean(gen_losses),
        'disc_loss': np.mean(disc_losses),
        'div_loss': np.mean(div_losses),
        'intensity_loss': np.mean(intensity_losses),
        'aux_reg_loss': np.mean(aux_reg_losses),
        'router_loss': router_loss.item(),
        'expert_distribution_loss': expert_distribution_loss.item(),
        'differentiation_loss': differentiation_loss.item(),
        'expert_entropy_loss': expert_entropy_loss.item(),
        'adaptive_load_balancing_loss': alb_loss.item(),
        'gan_loss': gan_loss_scaled.item(),
        **{f"gen_loss_{i}": gen_losses[i] for i in range(self.n_experts)},
        **{f"disc_loss_{i}": disc_losses[i] for i in range(self.n_experts)},
        **{f"div_loss_experts_{i}": div_losses[i] for i in range(self.n_experts)},
        **{f"intensity_loss_experts_{i}": intensity_losses[i] for i in range(self.n_experts)},
        **{f"aux_reg_loss_experts_{i}": aux_reg_losses[i] for i in range(self.n_experts)},
        **{f"std_intensities_experts_{i}": std_intensities_experts[i] for i in range(self.n_experts)},
        **{f"mean_intensities_experts_{i}": mean_intensities_experts[i] for i in range(self.n_experts)},
        **{f"n_choosen_experts_mean_epoch_{i}": class_counts[i] for i in range(self.n_experts)},
        }

        return log_metrics

    def discriminator_train_step(self, disc, fake_images, d_optimizer, criterion, class_counts, real_images, cond, batch_size) -> np.float32:
        """Returns Python float of disc_loss value"""
        # Train discriminator
        d_optimizer.zero_grad(set_to_none=True)

        # calculate loss for real images
        real_output, real_latent = disc(real_images, cond)
        real_labels = torch.ones_like(real_output)
        loss_real_disc = criterion(real_output, real_labels)

        # calculate loss for generated images
        fake_output, fake_latent = disc(fake_images.detach(), cond)
        fake_labels = torch.zeros_like(fake_output)
        loss_fake_disc = criterion(fake_output, fake_labels)

        # Accumulate and compute discriminator los
        disc_loss = loss_real_disc + loss_fake_disc
        disc_loss.backward()  # call backward computations on accumulated gradients for efficiency

        # d_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
        d_optimizer.step()
        return disc_loss

    def generator_train_step(self, fake_images, noise_1, generator, discriminator, a_reg, cond, g_optimizer,
                             a_optimizer, criterion, true_positions, std, intensity, class_counts, batch_size):
        # Train Generator
        g_optimizer.zero_grad(set_to_none=True)

        noise_2 = torch.randn(batch_size, self.noise_dim, device=cond.device)

        # generate fake images
        fake_images_2 = generator(noise_2, cond)

        # validate two images
        fake_output, fake_latent = discriminator(fake_images, cond)  # don't detach, so gradients flow back to generator
        fake_output_2, fake_latent_2 = discriminator(fake_images_2, cond)

        gen_loss = criterion(fake_output, torch.ones_like(fake_output))

        div_loss = self.sdi_gan_regularization(fake_latent, fake_latent_2,
                                               noise_1, noise_2,
                                               std, generator.di_strength)

        intensity_loss, mean_intenisties, std_intensity, mean_intensity = self.intensity_regularization(fake_images,
                                                                                                   intensity,
                                                                                                   generator.in_strength)

        gen_loss = gen_loss + div_loss + intensity_loss

        # Train auxiliary regressor
        a_optimizer.zero_grad()
        generated_positions = a_reg(fake_images)

        aux_reg_loss = self.regressor_loss(true_positions, generated_positions)*self.cfg.model.aux_reg.strength
        gen_loss += aux_reg_loss

        gen_loss.backward(retain_graph=True)
        g_optimizer.param_groups[0]['lr'] = self.cfg.model.generator.lr_g * class_counts.clone().detach()
        g_optimizer.step()
        a_optimizer.param_groups[0]['lr'] = self.cfg.model.aux_reg.lr_a * class_counts.clone().detach()
        a_optimizer.step()

        # # Auxiliary loss (Gradients flow back to the generator to update it)
        # generated_positions = a_reg(fake_images)
        # aux_reg_loss = aux_reg_criterion(true_positions, generated_positions, scaler_poz, AUX_STRENGTH)
        #
        # # Combined loss for generator
        # total_gen_loss = gen_loss + aux_reg_loss.detach()  # Detach aux_loss from generator
        #
        # # Backpropagate generator losses
        # total_gen_loss.backward()
        # g_optimizer.param_groups[0]['lr'] = LR_G * class_counts.clone().detach()
        # g_optimizer.step()
        #
        # # Train aux_reg separately
        # a_optimizer.zero_grad()
        # a_optimizer.param_groups[0]['lr'] = LR_A * class_counts.clone().detach()
        # a_optimizer.step()

        aux_reg_features = a_reg.feature_extractor(fake_images)
        return gen_loss, div_loss.data, intensity_loss.data, aux_reg_loss.data, std_intensity, mean_intensity,\
               mean_intenisties, aux_reg_features

    @staticmethod
    def sdi_gan_regularization(fake_latent, fake_latent_2, noise, noise_2, std, DI_STRENGTH):
        # Calculate the absolute differences and their means along the batch dimension
        abs_diff_latent = torch.mean(torch.abs(fake_latent - fake_latent_2), dim=1)
        abs_diff_noise = torch.mean(torch.abs(noise - noise_2), dim=1)

        # Compute the division term
        div = abs_diff_latent / (abs_diff_noise + 1e-5)

        # Calculate the div_loss
        div_loss = std * DI_STRENGTH / (div + 1e-5)

        # Calculate the final div_loss
        div_loss = torch.mean(std) * torch.mean(div_loss)

        return div_loss

    @staticmethod
    def intensity_regularization(gen_im_proton, intensity_proton, IN_STRENGTH):
        """
        Computes the intensity regularization loss for generated images, returning the loss, the sum of intensities per image,
        and the mean and standard deviation of the intensity across the batch.

        Args:
            gen_im_proton (torch.Tensor): A tensor of generated images with shape [batch_size, channels, height, width].
            intensity_proton (torch.Tensor): A tensor representing the target intensity values for the batch, with shape [batch_size].
            IN_STRENGTH (float): A scalar that controls the strength of the intensity regularization in the final loss.

        Returns:
            torch.Tensor: The intensity regularization loss, calculated as the Mean Absolute Error (MAE) between the scaled
                          sum of the intensities in the generated images and the target intensities, multiplied by `IN_STRENGTH`.
            torch.Tensor: The sum of intensities in each generated image, with shape [n_samples, 1].
            torch.Tensor: The standard deviation of the scaled intensity values across the batch.
            torch.Tensor: The mean of the scaled intensity values across the batch.
        """

        # Sum the intensities in the generated images
        # gen_im_proton_rescaled = torch.exp(gen_im_proton.clone().detach()) - 1 #<- this fixed previous bad optimization
        gen_im_proton_rescaled = torch.exp(gen_im_proton) - 1
        # print("Gen shape from model", gen_im_proton_rescaled.shape)
        # Gen shape from model torch.Size([138, 1, 56, 30])
        # After sum: torch.Size([138, 1, 1, 1])
        sum_all_axes_p_rescaled = torch.sum(gen_im_proton_rescaled, dim=[2, 3],
                                            keepdim=False)  # Sum along all but batch dimension

        # print(sum_all_axes_p_rescaled.shape)  # (batch_size_current, 1)
        # print(sum_all_axes_p_rescaled)
        # REMOVE THIS RESHAPE BECAUSE IT FLATTENS THE DATA FROM ALL EXPERTS
        # sum_all_axes_p_rescaled = sum_all_axes_p_rescaled.reshape(-1, 1)  # Scale and reshape back to (batch_size, 1)

        # Compute mean and std as PyTorch tensors
        std_intensity_scaled = sum_all_axes_p_rescaled.std()
        mean_intensity_scaled = sum_all_axes_p_rescaled.mean()  # scalar
        # print('---------------')
        # print(mean_intensity_scaled.shape)
        # print('---------------')
        # # Ensure intensity_proton is correctly shaped and on the same device
        intensity_proton = intensity_proton.view(-1, 1).to(
            gen_im_proton.device)  # Ensure it is of shape [batch_size, 1]

        # apply the MASK AS WELL FOR EXPERT COMPUTATIONS TO BOTH THE GENERATED AND REAL DATA
        # OR MAYBE CALCULATE THIS N_EXPERT times each for separate expert. TRY TO MAKE THIS PARALLEL

        # print('shape sum_all_axes_p_rescaled', sum_all_axes_p_rescaled.shape)
        # print('shape intensity_proton',intensity_proton.shape)
        assert sum_all_axes_p_rescaled.shape == intensity_proton.shape
        # Calculate MAE loss
        mae_value_p = F.l1_loss(sum_all_axes_p_rescaled, intensity_proton) * IN_STRENGTH

        return mae_value_p, sum_all_axes_p_rescaled, std_intensity_scaled, mean_intensity_scaled

    @staticmethod
    def regressor_loss(real_coords, fake_coords):
        # Ensure real_coords and fake_coords are on the same device
        # real_coords = real_coords.to(fake_coords.device)

        # Use in-place scaling if the scaler provides the scale and mean attributes
        # scale = torch.tensor(scaler_poz.scale_, device=fake_coords.device, dtype=torch.float32)
        # mean = torch.tensor(scaler_poz.mean_, device=fake_coords.device, dtype=torch.float32)
        #
        # # Scale fake_coords directly using PyTorch operations
        # fake_coords_scaled = (fake_coords - mean) / scale

        # Compute the MAE loss
        return F.mse_loss(fake_coords, real_coords)

    def evaluate(self, epoch, y_test, x_test, true_positions, std, intensity, cfg, device):
        ch_org = np.exp(x_test) - 1  # Original channels
        ch_org = np.array(ch_org).reshape(-1, *cfg.dataset.input_image_shape)
        ch_org = pd.DataFrame(sum_channels_parallel(ch_org)).values

        y_test_temp = torch.tensor(y_test, device=device)

        gates, logits = self.router(y_test_temp)

        _, predicted_expert = torch.max(gates, 1)

        indices_experts = [np.where(predicted_expert.cpu().numpy() == i)[0] for i in range(self.n_experts)]

        # Process expert indices dynamically
        ch_org_experts = []
        for i in range(self.n_experts):
            if len(indices_experts[i]) > 0:
                org = np.exp(x_test[indices_experts[i]]) - 1
                ch_org_exp = np.array(org).reshape(-1, *self.image_shape)
                del org
                ch_org_exp = pd.DataFrame(sum_channels_parallel(ch_org_exp)).values
            else:
                ch_org_exp = np.zeros((len(indices_experts[i]), 5))
            ch_org_experts.append(ch_org_exp)

        y_test_experts = [y_test[indices_experts[i]] for i in range(self.n_experts)]
        # Calculate WS distance across all distribution
        ws_mean, ws_std, ws_mean_exp, ws_std_exp = calculate_joint_ws_across_experts(
            min(epoch // 5 + 1, 5),
            [x_test[indices_experts[i]] for i in range(self.n_experts)],
            y_test_experts,
            self.generators, ch_org,
            ch_org_experts,
            self.noise_dim, device,
            n_experts=self.n_experts)

        # Generate Plots
        plot_experts = [None] * self.n_experts
        IDX_GENERATE = [1, 2, 3, 4, 5, 6]
        noise_cond_experts = [
            y_test[indices_experts[i]][IDX_GENERATE]
            if len(y_test[indices_experts[i]]) > len(IDX_GENERATE)
            else None
            for i in range(self.n_experts)
        ]

        # Log to WandB tool
        log_data = {
            'ws_mean': ws_mean,
            **{f"ws_mean_{i}": ws_mean_exp[i] for i in range(self.n_experts)},
            'ws_std': ws_std,
            **{f"ws_std_{i}": ws_std_exp[i] for i in range(self.n_experts)},
            'epoch': epoch
        }
        for i in range(self.n_experts):
            log_data[f"ws_mean_{i}"] = ws_mean_exp[i]
            log_data[f"ws_std_{i}"] = ws_std_exp[i]
            if cfg.wandb.plot_images:
                noise = torch.randn(len(IDX_GENERATE), self.noise_dim,
                                    device=device)  # same noise vector for each expert
                for i in range(self.n_experts):
                    plot = generate_and_save_images(self.generators[i], epoch, noise, noise_cond_experts[i],
                                                    x_test[indices_experts[i]],
                                                    0, 0, device,
                                                    f'Expert {i}')
                    plot_experts[i] = plot

                log_data[f"plot_expert_{i}"] = wandb.Image(plot_experts[i]) if not plot_experts[i] is None else None

        return log_data

    def get_expert_assignment_counts(self, expert_assignments: torch.Tensor) -> torch.Tensor:
        """Get count of samples assigned to each expert."""
        class_counts = torch.zeros(self.n_experts, dtype=torch.float, device=expert_assignments.device)
        for expert_idx in range(self.n_experts):
            class_counts[expert_idx] = (expert_assignments == expert_idx).sum().item()
        return class_counts / expert_assignments.size(0)


#
##
### OLD OLD OLD
##
#

    # def forward(self, z: torch.Tensor, cond: torch.Tensor, mode: str = 'generate'):
    #     """
    #     Forward pass through MoE system.
    #
    #     Args:
    #         z: Input tensor (noise for generation, images for discrimination)
    #         cond: Condition tensor
    #         mode: 'generate' or 'discriminate'
    #
    #         # generators output images of shape (batch_size, 1, 56, 30), but are squeezed to (batch_size, n_experts, 56, 30)
    #
    #     Returns:
    #         Generated images of shape (batch_size, n_experts, 56, 30).
    #         Discriminator outputs of shape (batch_size, n_experts).
    #         Gating scores of shape (batch_size, n_experts) and experts assignments.
    #         Auxiliary regressor outputs.
    #         Tuple of (outputs, expert_assignments, gates)
    #     """
    #     batch_size = cond.shape[0]
    #     device = cond.device
    #
    #     # Get expert assignments from router
    #     gates = self.router(cond)  # Shape: (batch_size, n_experts)
    #     _, expert_assignments = torch.max(gates, 1)  # Shape: (batch_size,)
    #
    #     if mode == 'generate':
    #         # Initialize output tensor - adjust dimensions based on your image size
    #         # shape = (batch_size, 1) + self.image_shape
    #         generated_images = torch.zeros(
    #             batch_size, self.n_experts, 56, 30,  # Assuming height=56, width=30
    #             device=device
    #         )
    #         # Process each expert's assigned samples
    #         for expert_idx in range(self.n_experts):
    #             expert_mask = (expert_assignments == expert_idx)
    #             expert_indices = expert_mask.nonzero(as_tuple=True)[0]
    #
    #             if expert_indices.numel() > 0:
    #                 expert_z = z[expert_indices]
    #                 expert_cond = cond[expert_indices]
    #
    #                 # Generate images for this expert (shape: [expert_batch, 1, 56, 30])
    #                 expert_generated = self.generators[expert_idx](expert_z, expert_cond)
    #
    #                 # Squeeze channel dim and place in the sparse tensor
    #                 generated_images[expert_indices, expert_idx] = expert_generated.squeeze(1)
    #
    #         return generated_images, expert_assignments, gates
    #
    #     elif mode == 'discriminate':
    #         disc_outputs = []
    #         disc_latents = []
    #
    #         for expert_idx in range(self.n_experts):
    #             expert_mask = (expert_assignments == expert_idx)
    #             expert_indices = expert_mask.nonzero(as_tuple=True)[0]
    #
    #             if expert_indices.numel() > 0:
    #                 expert_real = z[expert_indices]  # z is real images in this mode
    #                 expert_cond = cond[expert_indices]
    #                 disc_out, disc_latent = self.discriminators[expert_idx](expert_real, expert_cond)
    #                 disc_outputs.append((expert_indices, disc_out, disc_latent))
    #             else:
    #                 disc_outputs.append((expert_indices, None, None))
    #
    #         return disc_outputs, expert_assignments, gates
    #
    #     else:
    #         raise ValueError(f"Unknown mode: {mode}")