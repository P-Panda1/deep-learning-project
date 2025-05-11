import torch
import torch.nn as nn

# --- Encoder ---


class AudioEncoder(nn.Module):
    def __init__(self, input_channels=1, conv_channels=32, transformer_dim=256, num_layers=4):
        super(AudioEncoder, self).__init__()
        self.conv = nn.Sequential(
            # [B, 441000] → [B, 1, 1000]
            nn.MaxPool1d(kernel_size=441, stride=441),
            nn.Conv1d(input_channels, conv_channels, kernel_size=10,
                      stride=10, padding=0),  # [B, 1, 1000] → [B, 32, 100]
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Conv1d(conv_channels, 2 * conv_channels, kernel_size=12,
                      stride=4, padding=4),   # [B, 32, 100] → [B, 32, 25]
            nn.BatchNorm1d(2 * conv_channels),
            nn.ReLU(),
            nn.Conv1d(2 * conv_channels, conv_channels, kernel_size=5,
                      stride=1, padding=2),   # [B, 32, 25] → [B, 32, 25]
        )

        # self.pos_encoder = nn.Parameter(torch.randn(1, 100, transformer_dim))  # Pos encoding
        # [B, 32, 25] → [B, 25, 256]
        self.linear_proj = nn.Linear(conv_channels, transformer_dim)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_dim, nhead=8)
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 441000] → [B, 1, 441000]
        x = torch.abs(x)
        x = self.conv(x)    # → [B, 256, 25]
        x = x.permute(0, 2, 1)  # → [B, 25, 256]
        x = self.linear_proj(x)  # → [B, 25, 256]
        # x = x + self.pos_encoder[:, :x.size(1)]  # Add positional encoding
        # x = x.permute(1, 0, 2)  # → [4410, B, 256] for Transformer
        # x = self.transformer(x)  # → [4410, B, 256]
        # x = x.permute(1, 0, 2)  # → [B, 4410, 256]
        # Commenting out the transformer for simplicity
        return x


class AudioDecoder(nn.Module):
    def __init__(self, transformer_dim=256, num_classes=10):
        super(AudioDecoder, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # → [B, 256, 1]
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 100, 256] → [B, 256, 100]
        x = self.pool(x).squeeze(-1)  # → [B, 256]
        x = self.classifier(x)  # → [B, 10]
        return x

    # --- Full Model ---


class ConvTransformerAudioClassifier(nn.Module):
    def __init__(self):
        super(ConvTransformerAudioClassifier, self).__init__()
        self.encoder = AudioEncoder()
        self.decoder = AudioDecoder()

    def forward(self, x):
        encoded = self.encoder(x)  # → [B, ~11025, 256]
        logits = self.decoder(encoded)  # → [B, 10]
        return logits
