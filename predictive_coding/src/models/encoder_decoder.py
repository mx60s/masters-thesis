from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .residual import DownBlock2d, UpBlock2d
from .attention import MultiHeadAttention, PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, in_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down1 = DownBlock2d(64, 64, layers[0])
        self.down2 = DownBlock2d(64, 128, layers[1], downsample=True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)  # 32
        x = self.bn1(x)
        x = F.relu(x)

        x = self.maxpool(x)  # 16

        x = self.down1(x)  # 16
        x = self.down2(x)  # 8

        return x


class BiggerDecoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(128, out_channels, kernel_size=1, stride=1)
        self.up1 = UpBlock2d(256, 128, layers[0], upsample=True)
        self.up2 = UpBlock2d(128, 128, layers[1], upsample=True)
        self.up3 = UpBlock2d(128, 128, 1, upsample=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None)
        x = self.up2(x, None)
        x = self.up3(x, None)
        x = self.conv1(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(64, out_channels, kernel_size=1, stride=1)
        self.up1 = UpBlock2d(128, 64, layers[0], upsample=True)
        self.up2 = UpBlock2d(64, 64, layers[1], upsample=True)
        self.up3 = UpBlock2d(64, 64, 1, upsample=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x, None)
        x = self.up2(x, None)
        x = self.up3(x, None)

        x = self.conv1(x)

        return x


class FFN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.reshape(B, L, C, H, W)

        return x

class BottleneckCoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: List[int], seq_len: int, num_actions:int = None, bottleneck_dim=32) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = Decoder(in_channels, out_channels, layers[::-1])
        self.attention_1 = MultiHeadAttention(128, num_heads=8)
        self.ffn_1 = FFN(128, 128)
        self.attention_2 = MultiHeadAttention(128, num_heads=8)
        self.ffn_2 = FFN(128, 128)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.bottleneck = NewBottleneck(actions=num_actions, in_channels=128, out_channels=128, bottleneck_dim=bottleneck_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        actions = actions.reshape(B*L, -1)

        f = self.encoder(x)
        f = self.bottleneck(f, actions)

        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        residual = f
        f = self.attention_1(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.attention_2(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])

        x = self.decoder(f)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
            actions = actions.reshape(B*L, -1)
    
            f = self.encoder(x)
            f = self.bottleneck(f, actions)
            
            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])
    
            if which == 0:
                return f
    
            residual = f
            f = self.attention_1(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_1(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            if which == 1:
                return f
    
            residual = f
            f = self.attention_2(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_2(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            return f
    

class VestibularCoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: List[int], seq_len: int, num_actions:int = None) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = BiggerDecoder(in_channels * 2, out_channels, layers[::-1])
        self.attention_1 = MultiHeadAttention(128 * 2, num_heads=8)
        self.ffn_1 = FFN(128 * 2, 128 * 2)
        self.attention_2 = MultiHeadAttention(128 * 2, num_heads=8)
        self.ffn_2 = FFN(128 * 2, 128 * 2)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.bottleneck = ConvBottleneck(actions=num_actions, in_channels=128, out_channels=128, features=128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)
        actions = actions.reshape(B*L, -1)

        f = self.encoder(x)
        f = self.bottleneck(f, actions)

        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        residual = f
        f = self.attention_1(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.attention_2(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])

        x = self.decoder(f)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
    
            f = self.encoder(x)
            f = self.bottleneck(f, actions)
            
            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])
    
            if which == 0:
                return f
    
            residual = f
            f = self.attention_1(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_1(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            if which == 1:
                return f
    
            residual = f
            f = self.attention_2(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_2(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            return f

    def get_bottleneck_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
    
            f = self.encoder(x)
            f = self.bottleneck.get_latents(f, actions)

            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

            return f

class PredictiveCoder(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, layers: List[int], seq_len: int
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = Decoder(in_channels, out_channels, layers[::-1])
        self.attention_1 = MultiHeadAttention(128, num_heads=8)
        self.ffn_1 = FFN(128, 128)
        self.attention_2 = MultiHeadAttention(128, num_heads=8)
        self.ffn_2 = FFN(128, 128)
        self.mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        #self.position = PositionalEncoding(512, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C, H, W = x.shape
        x = x.reshape(B * L, C, H, W)

        f = self.encoder(x)

        f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])

        residual = f
        f = self.attention_1(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_1(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.attention_2(f, f, f, self.mask)[0]
        f = f + residual
        f = F.layer_norm(f, f.shape[1:])

        residual = f
        f = self.ffn_2(f) + residual
        f = F.layer_norm(f, f.shape[1:])

        f = f.reshape(B * L, f.shape[2], f.shape[3], f.shape[4])

        x = self.decoder(f)
        x = x.reshape(B, L, C, H, W)

        return x

    def get_latents(self, x: torch.Tensor, actions: torch.Tensor = None, which=-1) -> torch.Tensor:
        with torch.no_grad():
            B, L, C, H, W = x.shape
            x = x.reshape(B * L, C, H, W)
    
            f = self.encoder(x)
            
            f = f.reshape(B, L, f.shape[1], f.shape[2], f.shape[3])
    
            if which == 0:
                return f
    
            residual = f
            f = self.attention_1(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_1(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            if which == 1:
                return f
    
            residual = f
            f = self.attention_2(f, f, f, self.mask)[0]
            f = f + residual
            f = F.layer_norm(f, f.shape[1:])
    
            residual = f
            f = self.ffn_2(f) + residual
            f = F.layer_norm(f, f.shape[1:])
    
            return f

class Autoencoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, layers)
        self.decoder = Decoder(in_channels, out_channels, layers[::-1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)

        return x

    def get_latents(self, x, actions: torch.Tensor = None):
        return self.encoder(x)

# ConvBottleneck(actions=num_actions, in_channels=128, out_channels=128, features=128)
# torch.Size([640, 128, 8, 8])
class ConvBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, features: int=128):
        super().__init__()
        self.features = features
        self.linear = nn.Linear(actions, self.features * 8 * 8, bias=False) #512
        self.conv = nn.Conv2d(self.features + in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, features, actions):
        B, C, H, W = features.shape
        conv_actions = self.linear(actions).reshape(B, self.features, H, W)

        output = torch.cat([features, conv_actions], dim=1)
        
        #output = self.conv(output)
        #output = self.bn(output)
        #output = F.relu(output)

        return output

    def get_latents(self, features, actions):
        B, C, H, W = features.shape
        return self.linear(actions).reshape(B, self.features, H, W)


class NewBottleneck(nn.Module):
    def __init__(self, actions: int, in_channels: int, out_channels: int, bottleneck_dim: int = 32):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_dim)
        self.action_embed = nn.Linear(actions, bottleneck_dim, bias=False)
        self.combine = nn.Linear(bottleneck_dim * 2, bottleneck_dim, bias=False)
        self.expand = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, features, actions):
        B, C, H, W = features.shape

        latent = self.compress(features)
        latent = self.bn1(latent)
        latent = F.relu(latent)

        latent_flat = latent.view(B, -1, H*W).mean(dim=2)
        
        action_embed = self.action_embed(actions)
        combined = torch.cat([latent_flat, action_embed], dim=1)
        combined = self.combine(combined)
        combined = F.relu(combined)
        
        expanded = combined.view(B, -1, 1, 1).expand(-1, -1, H, W)
        
        output = self.expand(expanded)
        output = self.bn2(output)
        output = F.relu(output)
        
        return output#, latent, expanded