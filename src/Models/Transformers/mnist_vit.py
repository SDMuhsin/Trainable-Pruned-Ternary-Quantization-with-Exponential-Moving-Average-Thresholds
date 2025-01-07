import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size=4, d_model=256):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.projection(x)  # (B, d_model, H', W')
        x = x.flatten(2)        # (B, d_model, N)
        x = x.transpose(1, 2)   # (B, N, d_model)
        return x

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, d_model):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_position_embeddings, d_model))
        
    def forward(self, x):
        return x + self.position_embeddings[:, :x.size(1), :]

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, d_model):
        super().__init__()
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_position_embeddings, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class VisionTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        nhead,
        d_hid,
        nlayers,
        dropout,
        nb_features_projection,
        d_model,
        num_classes,
        classification_pool='cls',  # 'cls' or 'mean'
        n_conv_layers=1,           # number of conv layers before transformer
        pos_encoder_type='learned' # 'learned' or 'sinusoidal'
    ):
        super().__init__()
        
        # Image size for MNIST is typically 28x28
        self.patch_embed = PatchEmbedding(in_channels, patch_size=4, d_model=d_model)
        self.num_patches = (28 // 4) * (28 // 4)  # For MNIST with patch_size=4
        self.classification_pool = classification_pool
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Position encoding
        max_position_embeddings = self.num_patches + 1  # +1 for cls token
        if pos_encoder_type == 'learned':
            self.pos_encoder = LearnedPositionalEncoding(max_position_embeddings, d_model)
        else:
            self.pos_encoder = SinusoidalPositionalEncoding(max_position_embeddings, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        
        # Final classification head
        self.fc = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        if isinstance(self.pos_encoder, LearnedPositionalEncoding):
            nn.init.normal_(self.pos_encoder.position_embeddings, std=0.02)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, d_model)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        
        # Pool according to strategy
        if self.classification_pool == 'cls':
            x = x[:, 0]  # Use CLS token
        else:  # 'mean'
            x = x.mean(dim=1)  # Mean of all tokens
        
        # Classification head
        x = self.fc(x)
        
        # Output
        output = F.log_softmax(x, dim=-1)
        return output

if __name__ == '__main__':
    # Example usage
    device = torch.device("cpu")
    model = VisionTransformer(
        in_channels=1,
        nhead=8,
        d_hid=384,
        nlayers=6,
        dropout=0.1,
        nb_features_projection=64,
        d_model=256,
        num_classes=10,
        classification_pool='cls',
        n_conv_layers=1,
        pos_encoder_type='learned'
    )
    model.float()
    model = model.to(device)
    
    # Test with dummy input
    x = torch.randn(2, 1, 28, 28)  # (batch_size, channels, height, width)
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (2, 10)
