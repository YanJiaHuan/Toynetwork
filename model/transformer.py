import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Vanilla Transformer model
class VanillaTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_classes=10):
        super(VanillaTransformer, self).__init__()

        # Embedding layer for input tokens
        self.embedding = nn.Linear(32, embed_dim)  # Assume input dimension is 32

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # Project input to embedding dimension

        # Transformer expects shape (seq_len, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)

        # Take the representation of the first token (class token in NLP analogy)
        x = x[0]  # Shape: (batch_size, embed_dim)
        x = self.fc(x)
        return x

# Example Input
input_data = torch.randn(8, 10, 32)  # (batch_size, seq_len, input_dim)
vanilla_transformer = VanillaTransformer()
vanilla_output = vanilla_transformer(input_data)
print('output',vanilla_output)  # Output shape: (batch_size, num_classes)