"""
atlas.model
-----------
β-VAE architecture for cross-species behavioural embedding.

Species-specific Conv1d encoders feed into a shared latent space;
species-specific MLP decoders reconstruct the input window.
A shared TempHead regresses normalised temperature from z.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeciesEncoder(nn.Module):
    """Conv1d encoder: (B, T) → (B, 2*latent_dim).

    ``AdaptiveAvgPool1d(4)`` collapses any window length T to 4 time-steps,
    so the same architecture works for jellyfish (60 s) and fish (5 s).
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),   # always → 4 steps regardless of input T
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 128), nn.ReLU(),
            nn.Linear(128, 2 * latent_dim),
        )

    def forward(self, x):   # x: (B, T)
        return self.fc(self.conv(x.unsqueeze(1)).flatten(1))


class SpeciesDecoder(nn.Module):
    """MLP decoder: latent_dim → seq_len (species-specific)."""

    def __init__(self, latent_dim, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, seq_len),
        )

    def forward(self, z):
        return self.net(z)


class TempHead(nn.Module):
    """Shared regression head: z → predicted normalised temperature."""

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)   # (B,)


class BehavioralVAE(nn.Module):
    """β-VAE with species-specific encoders/decoders and a shared latent space.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the shared latent space (default 8).
    jelly_seq : int
        Jellyfish window length in seconds (= samples at 1 FPS).
    fish_seq : int
        Stickleback window length in seconds.
    """

    def __init__(self, latent_dim=8, jelly_seq=60, fish_seq=5):
        super().__init__()
        self.jelly_enc  = SpeciesEncoder(latent_dim)
        self.fish_enc   = SpeciesEncoder(latent_dim)
        self.jelly_dec  = SpeciesDecoder(latent_dim, jelly_seq)
        self.fish_dec   = SpeciesDecoder(latent_dim, fish_seq)
        self.temp_head  = TempHead(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    def forward(self, x, species):
        """Forward pass for one species.

        Parameters
        ----------
        x : torch.Tensor of shape (B, T)
        species : ``'jellyfish'`` | ``'fish'``

        Returns
        -------
        recon : (B, T)
        mu : (B, latent_dim)
        logvar : (B, latent_dim)
        z : (B, latent_dim)
        temp_pred : (B,)
        """
        enc = self.jelly_enc if species == 'jellyfish' else self.fish_enc
        dec = self.jelly_dec if species == 'jellyfish' else self.fish_dec
        mu, logvar = enc(x).chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return dec(z), mu, logvar, z, self.temp_head(z)
