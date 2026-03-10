"""
atlas.train
-----------
Dataset class, ELBO / temperature-regression losses, and the training loop
for BehavioralVAE.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SpeedDataset(Dataset):
    """PyTorch Dataset wrapping normalised speed windows and temperature labels.

    Parameters
    ----------
    windows : np.ndarray of shape (N, T, 1)
        z-scored speed windows (the trailing 1-channel dimension is stripped).
    temps_z : np.ndarray of shape (N,)
        z-scored temperature labels (NaN allowed).
    """

    def __init__(self, windows, temps_z):
        self.x    = torch.tensor(windows[:, :, 0], dtype=torch.float32)
        self.temp = torch.tensor(temps_z,           dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.temp[i]


def elbo(recon, x, mu, logvar, beta):
    """ELBO = reconstruction MSE + β × KL divergence.

    Returns
    -------
    loss : torch.Tensor  (scalar, for backward)
    recon_val : float
    kl_val : float
    """
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss.item(), kl.item()


def temp_loss(pred, target):
    """MSE temperature regression loss, ignoring NaN targets.

    Returns
    -------
    torch.Tensor (scalar)
    """
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.mse_loss(pred[mask], target[mask])


def train(model, jelly_z, jelly_temps_z, fish_z, fish_temps_z,
          n_epochs=250, n_anneal=40, beta=4.0, lambda_temp=2.0,
          batch=64, lr=1e-3, log_every=25, device=None):
    """Train BehavioralVAE with interleaved jellyfish / fish mini-batches.

    Parameters
    ----------
    model : BehavioralVAE
    jelly_z : np.ndarray (N_j, JELLY_WINDOW, 1)
    jelly_temps_z : np.ndarray (N_j,)
    fish_z : np.ndarray (N_f, FISH_WINDOW, 1)
    fish_temps_z : np.ndarray (N_f,)
    n_epochs : int
    n_anneal : int  — epochs over which β is linearly warmed up from 0
    beta : float    — final β value
    lambda_temp : float  — weight on temperature regression loss
    batch : int
    lr : float
    log_every : int  — print every N epochs
    device : str | None  — defaults to 'cuda' if available else 'cpu'

    Returns
    -------
    history : np.ndarray of shape (n_epochs, 6)
        Columns: epoch, beta_t, total_loss, recon_loss, kl_loss, temp_loss
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    jelly_ds     = SpeedDataset(jelly_z, jelly_temps_z)
    fish_ds      = SpeedDataset(fish_z,  fish_temps_z)
    jelly_loader = DataLoader(jelly_ds, batch_size=batch, shuffle=True,  drop_last=True)
    fish_loader  = DataLoader(fish_ds,  batch_size=batch, shuffle=True,  drop_last=True)

    history = []

    for epoch in range(1, n_epochs + 1):
        beta_t = min(beta, beta * epoch / n_anneal)
        model.train()
        totals = [0.0] * 4
        n = 0
        fish_iter = iter(fish_loader)

        for jx, jt in jelly_loader:
            jx, jt = jx.to(device), jt.to(device)
            recon_j, mu_j, lv_j, _, tp_j = model(jx, 'jellyfish')
            loss_j, rl_j, kl_j = elbo(recon_j, jx, mu_j, lv_j, beta_t)
            tl_j = temp_loss(tp_j, jt)

            try:
                fx, ft = next(fish_iter)
            except StopIteration:
                fish_iter = iter(fish_loader)
                fx, ft   = next(fish_iter)
            fx, ft = fx.to(device), ft.to(device)
            recon_f, mu_f, lv_f, _, tp_f = model(fx, 'fish')
            loss_f, rl_f, kl_f = elbo(recon_f, fx, mu_f, lv_f, beta_t)
            tl_f = temp_loss(tp_f, ft)

            loss = loss_j + loss_f + lambda_temp * (tl_j + tl_f)
            opt.zero_grad()
            loss.backward()
            opt.step()

            totals[0] += loss.item()
            totals[1] += rl_j + rl_f
            totals[2] += kl_j + kl_f
            totals[3] += tl_j.item() + tl_f.item()
            n += 1

        row = [epoch, beta_t] + [v / n for v in totals]
        history.append(row)
        if epoch % log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{n_epochs}  β={beta_t:.2f}  "
                  f"loss={row[2]:.4f}  recon={row[3]:.4f}  "
                  f"KL={row[4]:.4f}  temp={row[5]:.4f}")

    print("\nTraining complete.")
    return np.array(history)


def encode_all(model, windows, species, device=None, batch=256):
    """Encode a batch of speed windows → latent means.

    Parameters
    ----------
    model : BehavioralVAE (eval mode is set internally)
    windows : np.ndarray of shape (N, T)  — z-scored, no channel dim
    species : ``'jellyfish'`` | ``'fish'``
    device : str | None
    batch : int

    Returns
    -------
    mu : np.ndarray of shape (N, latent_dim)
    temp_pred : np.ndarray of shape (N,)  — normalised predicted temperature
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    enc = model.jelly_enc if species == 'jellyfish' else model.fish_enc
    mus, temps_pred = [], []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            x  = torch.tensor(windows[i:i+batch], dtype=torch.float32).to(device)
            mu, _ = enc(x).chunk(2, dim=-1)
            tp    = model.temp_head(mu)
            mus.append(mu.cpu().numpy())
            temps_pred.append(tp.cpu().numpy())
    return np.vstack(mus), np.concatenate(temps_pred)
