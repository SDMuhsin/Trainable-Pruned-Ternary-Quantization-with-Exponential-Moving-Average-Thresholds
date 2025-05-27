from math import floor

import numpy as np

import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F

# For Xavier Normal initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# Class defining the encoder (same as your provided code)
class MnistEncoderModel(nn.Module):
    def __init__(self, input_channels=1): # nb_classes removed as it's not used here
        super(MnistEncoderModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self, x):
        # Conv Block 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv Block 2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Output shape for MNIST (1, 28, 28) input: (batch_size, 20, 4, 4)
        return x

# Class defining the hybrid CNN-ViT classification model
class HybridCNNViTModel(nn.Module):
    def __init__(self, input_channels=1, nb_classes=10,
                 cnn_output_channels=20, cnn_output_spatial_dim=4, # From MnistEncoderModel
                 vit_embed_dim=None, vit_num_heads=4, vit_num_layers=3,
                 vit_mlp_dim=80, vit_dropout=0.1):
        super(HybridCNNViTModel, self).__init__()

        # CNN Encoder
        self.encoder = MnistEncoderModel(input_channels)

        # Determine ViT embedding dimension if not specified
        # Defaults to the number of channels from the CNN encoder
        if vit_embed_dim is None:
            vit_embed_dim = cnn_output_channels

        # Linear projection if CNN output channels don't match ViT embed_dim
        # For this setup, cnn_output_channels (20) will be our vit_embed_dim.
        if cnn_output_channels != vit_embed_dim:
            self.projection = nn.Conv2d(cnn_output_channels, vit_embed_dim, kernel_size=1) # 1x1 conv for projection
        else:
            self.projection = nn.Identity()


        self.vit_embed_dim = vit_embed_dim
        num_patches = cnn_output_spatial_dim * cnn_output_spatial_dim # 4*4 = 16 for MNIST

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vit_embed_dim))
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, self.vit_embed_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vit_embed_dim,
            nhead=vit_num_heads,
            dim_feedforward=vit_mlp_dim,
            dropout=vit_dropout,
            activation=F.relu,
            batch_first=True, # Expects (batch, seq, feature)
            norm_first=True # Common in modern ViTs
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=vit_num_layers
        )

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.vit_embed_dim),
            nn.Linear(self.vit_embed_dim, nb_classes)
        )

        # Dropout for the final classification (similar to fc_drop in original)
        self.fc_drop = nn.Dropout(vit_dropout if vit_dropout > 0 else 0.0) # Use vit_dropout or specify

        # Initialize weights
        self.apply(weights_init)
        # Explicitly initialize CLS token and positional embeddings (often done with truncated normal)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)


    def forward(self, x):
        # Encoding
        x = self.encoder(x)  # Output: (batch_size, cnn_output_channels, H, W), e.g. (B, 20, 4, 4)

        # Optional projection
        x = self.projection(x) # Output: (batch_size, vit_embed_dim, H, W)

        # Reshape for ViT: (batch_size, num_patches, embed_dim)
        # (B, D, H, W) -> (B, D, N) -> (B, N, D) where N = H*W
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, D) e.g. (B, 16, 20)

        # Prepend CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D) e.g. (B, 17, 20)

        # Add positional embedding
        x = x + self.pos_embedding  # (B, N+1, D)

        # Transformer Encoder
        x = self.transformer_encoder(x)  # (B, N+1, D)

        # Get CLS token output for classification
        cls_output = x[:, 0]  # (B, D)

        # Classification head
        cls_output = self.fc_drop(cls_output) # Apply dropout
        output_logits = self.mlp_head(cls_output) # (B, nb_classes)

        # Output (log_softmax for compatibility with NLLLoss)
        output = F.log_softmax(output_logits, dim=-1)

        return output

if __name__=='__main__':
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Original Model Summary (for comparison) ---
    print("--- Original MnistClassificationModel ---")
    # For the original model summary to work without error with 28x28 input,
    # the fc1 layer needs to accept 320 inputs, not 80.
    # We'll define a corrected version for accurate comparison here.
    class CorrectedMnistClassificationModel(nn.Module):
        def __init__(self, input_channels=1, nb_classes=10):
            super(CorrectedMnistClassificationModel, self).__init__()
            self.encoder = MnistEncoderModel(input_channels) # Output 20*4*4 = 320 features
            self.fc1 = nn.Linear(320, 50) # Corrected input size
            self.fc2 = nn.Linear(50, nb_classes)
            self.fc_drop = nn.Dropout()

        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1) # Flatten
            x = F.relu(self.fc1(x))
            x = self.fc_drop(x)
            x = self.fc2(x)
            return F.log_softmax(x, dim=-1)

    original_model = CorrectedMnistClassificationModel(input_channels=1, nb_classes=10)
    original_model = original_model.to(device)
    original_model.apply(weights_init) # Apply initialization

    # Summary of the original model (corrected)
    # Note: torchinfo expects batch_size in input_size
    print(summary(original_model, input_size=(1, 1, 28, 28), verbose=0))
    print("\n")


    # --- Hybrid CNN-ViT Model ---
    print("--- HybridCNNViTModel ---")
    # Parameters for the ViT part to be roughly similar to the original FC layers:
    # Original FCs (320->50, 50->10): (320*50+50) + (50*10+10) = 16050 + 510 = 16560 params.
    # Hybrid model's ViT:
    #   CNN encoder: MnistEncoderModel (20 output channels, 4x4 spatial output)
    #   ViT config: embed_dim=20 (from CNN), num_heads=4, num_layers=3, mlp_dim=80
    #   CLS token: 20 params
    #   Positional embedding: (16+1)*20 = 340 params
    #   Transformer Layers (3 layers):
    #     Each layer (d_model=20, nhead=4, dim_feedforward=80, norm_first=True):
    #       Self-Attn (incl. in_proj, out_proj): (3*20*20 + 3*20) + (20*20 + 20) = 1260 + 420 = 1680
    #       MLP (2 linears): (20*80 + 80) + (80*20 + 20) = 1680 + 1620 = 3300
    #       LayerNorms (2 per layer): 2 * (2*20) = 80
    #       Total per layer: ~1680 + 3300 + 80 = 5060 params
    #     3 layers: 3 * 5060 = 15180 params
    #   MLP Head (LayerNorm + Linear): (2*20) + (20*10 + 10) = 40 + 210 = 250 params
    #   Total ViT part: 20 (cls) + 340 (pos) + 15180 (transformer) + 250 (head) = 15790 params.
    # This is very close to the 16560 params of the original FC layers.

    hybrid_model = HybridCNNViTModel(
        input_channels=1,
        nb_classes=10,
        cnn_output_channels=20,         # From MnistEncoderModel's conv2
        cnn_output_spatial_dim=4,       # From MnistEncoderModel's output H or W (4x4)
        vit_embed_dim=20,               # Dimension for ViT tokens, matches CNN output channels
        vit_num_heads=4,                # Number of attention heads (must divide vit_embed_dim)
        vit_num_layers=3,               # Number of transformer encoder layers
        vit_mlp_dim=80,                 # Dimension of the feed-forward network in transformer (e.g., 4*embed_dim)
        vit_dropout=0.1
    )
    hybrid_model.float() # Ensure model is float32
    hybrid_model = hybrid_model.to(device)

    # Summary of the hybrid model
    print(summary(hybrid_model, input_size=(1, 1, 28, 28), verbose=0))

    # Test forward pass
    print("\nTesting forward pass with random input...")
    dummy_input = torch.randn(2, 1, 28, 28).to(device) # Batch size of 2
    try:
        output = hybrid_model(dummy_input)
        print("Hybrid model forward pass successful.")
        print("Output shape:", output.shape) # Expected: (2, 10)

        original_output = original_model(dummy_input)
        print("Original model forward pass successful.")
        print("Output shape:", original_output.shape) # Expected: (2, 10)

    except Exception as e:
        print(f"Error during forward pass: {e}")
