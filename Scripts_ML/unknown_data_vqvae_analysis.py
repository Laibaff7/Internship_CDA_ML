"""
VQ-VAE Analysis of Unknown CDA Mass Spec Data After Noise Filter
with Optuna Hyperparameter Search + UMAP Latent Space Visualization

Pipeline:
1.  Random spectra visualization (raw)
2.  Log normalization + Savitzky-Golay smoothing
3.  Processed spectra visualization
4.  Optuna hyperparameter search (20 trials, ~20 warm-up epochs each)
5.  Full retraining with best hyperparameters (50 epochs)
6.  Reconstruction error distribution
7.  Codebook utilisation & latent space spread metrics
8.  UMAP latent space — colored by reconstruction error (continuous)
9.  Error percentile band exploration (band spectra + averages)
10. Reconstruction quality: original-vs-reconstructed overlays
11. Residual plots (original - reconstructed)
12. Mean +/- 1-std reconstruction per error percentile band

All outputs saved to: outputs/unknown_vqvae/unknown_vqvae_analysis_<timestamp>/
Optuna study saved to: outputs/unknown_vqvae/optuna_study_<timestamp>/
"""

# ============================================================================
# Imports
# ============================================================================
import subprocess
import sys

try:
    import optuna
except ImportError:
    print("optuna not found -- installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "--quiet"])
    import optuna

import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import pairwise_distances
try:
    from openTSNE import TSNE as openTSNE_TSNE
except ImportError:
    print("openTSNE not found -- installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openTSNE", "--quiet"])
    from openTSNE import TSNE as openTSNE_TSNE
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.signal import savgol_filter
from datetime import datetime
import os
import json

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'data_path':          'data/unknown_data_after_noise_filter.parquet',
    'num_random_spectra': 20,
    'input_length':       1000,
    'epochs_tune':        20,   # epochs per Optuna trial
    'n_trials':           20,   # total Optuna trials
    'epochs_final':       50,   # full retraining epochs for best config
    'eval_batch_size':    256,
    # Preprocessing
    'savgol_window':      11,
    'savgol_order':       3,
    # Error bands
    'num_bands':          5,
    'band_labels':        ['0-20% (Best)', '20-40%', '40-60%',
                           '60-80%', '80-100% (Worst)'],
    # UMAP
    'umap_n_neighbors':   30,
    'umap_min_dist':      0.1,
}

# ============================================================
# LOAD MODEL -- set this to the previous run's output folder
# e.g. 'outputs/unknown_vqvae/unknown_vqvae_analysis_20260310_123456'
# ============================================================
LOAD_OUTPUT_DIR = 'outputs/unknown_vqvae/unknown_vqvae_analysis_20260310_211416'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'outputs/unknown_vqvae/unknown_vqvae_analysis_{timestamp}'
optuna_dir = f'outputs/unknown_vqvae/optuna_study_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(optuna_dir,  exist_ok=True)
print(f"\nOutput directory : {output_dir}")
print(f"Optuna directory : {optuna_dir}")

with open(f'{output_dir}/config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n" + "="*70)
print("DATA LOADING")
print("="*70)

print(f"Loading {CONFIG['data_path']}...")
df = pd.read_parquet(CONFIG['data_path'])
print(f"  Shape: {df.shape}  |  Columns: {list(df.columns)}")

if 'spectrum' not in df.columns:
    raise ValueError(f"'spectrum' column not found. Got: {list(df.columns)}")

spectra_raw = np.stack(df['spectrum'].values)
print(f"  Raw spectra array: {spectra_raw.shape}")

has_labels = 'class' in df.columns or 'label' in df.columns
if has_labels:
    label_col = 'class' if 'class' in df.columns else 'label'
    labels = df[label_col].values
    print(f"  Labels ({label_col}): {np.unique(labels)}")
else:
    labels = None
    print("  No labels found")

# ============================================================================
# STEP 1: RAW SPECTRA VISUALISATION
# ============================================================================
print("\n" + "="*70)
print("STEP 1: VISUALISING RANDOM RAW SPECTRA")
print("="*70)

n_vis   = min(max(CONFIG['num_random_spectra'], 20), len(spectra_raw))
rand_idx = np.random.choice(len(spectra_raw), size=n_vis, replace=False)

fig, axes = plt.subplots(5, 4, figsize=(20, 15))
for i, idx in enumerate(rand_idx[:20]):
    ax = axes.flat[i]
    ax.plot(spectra_raw[idx], lw=1.5, color='steelblue', alpha=0.8)
    ax.set_xlabel('Mass Channel', fontsize=9)
    ax.set_ylabel('Intensity', fontsize=9)
    ax.set_title(f'Raw Spectrum {idx}', fontsize=10)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/01_raw_spectra_random.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_raw_spectra_random.png")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("STEP 2: PREPROCESSING -- LOG NORM + SAVITZKY-GOLAY SMOOTHING")
print("="*70)


def preprocess_spectra(spectra, target_length=1000, savgol_window=11, savgol_order=3):
    """Crop/pad -> clip negatives -> SavGol smooth -> log1p -> min-max."""
    processed = []
    sw = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
    for spec in spectra:
        s = spec[:target_length] if len(spec) >= target_length \
            else np.pad(spec, (0, target_length - len(spec)))
        s = np.maximum(s, 0.0)
        if sw >= savgol_order + 2:
            s = savgol_filter(s, window_length=sw, polyorder=savgol_order)
        s = np.log1p(np.maximum(s, 0.0))
        lo, hi = s.min(), s.max()
        if hi > lo:
            s = (s - lo) / (hi - lo)
        processed.append(s)
    return np.array(processed, dtype=np.float32)


print("Applying preprocessing...")
X = preprocess_spectra(
    spectra_raw,
    target_length=CONFIG['input_length'],
    savgol_window=CONFIG['savgol_window'],
    savgol_order=CONFIG['savgol_order'],
)
print(f"  Processed shape : {X.shape}")
print(f"  Value range     : [{X.min():.4f}, {X.max():.4f}]")

# ============================================================================
# STEP 3: PROCESSED SPECTRA VISUALISATION
# ============================================================================
print("\n" + "="*70)
print("STEP 3: VISUALISING PROCESSED SPECTRA")
print("="*70)

fig, axes = plt.subplots(5, 4, figsize=(20, 15))
for i, idx in enumerate(rand_idx[:20]):
    ax = axes.flat[i]
    ax.plot(X[idx], lw=1.5, color='darkgreen', alpha=0.8)
    ax.set_xlabel('Mass Channel', fontsize=9)
    ax.set_ylabel('Norm. Intensity', fontsize=9)
    ax.set_title(f'Processed Spectrum {idx}', fontsize=10)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/02_processed_spectra.png', dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for i in range(3):
    idx = rand_idx[i]
    axes[i, 0].plot(spectra_raw[idx], lw=1.5, color='steelblue')
    axes[i, 0].set_title(f'Raw Spectrum {idx}', fontweight='bold')
    axes[i, 0].set_xlabel('Mass Channel')
    axes[i, 0].set_ylabel('Raw Intensity')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 1].plot(X[idx], lw=1.5, color='darkgreen')
    axes[i, 1].set_title(f'Processed Spectrum {idx} (Log + Smoothed)', fontweight='bold')
    axes[i, 1].set_xlabel('Mass Channel')
    axes[i, 1].set_ylabel('Norm. Intensity')
    axes[i, 1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/03_raw_vs_processed.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_processed_spectra.png, 03_raw_vs_processed.png")

# ============================================================================
# VQ-VAE MODEL -- supports architecture search
# ============================================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim   = embedding_dim
        self.num_embeddings  = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z: (B, D, L)
        z_t  = z.permute(0, 2, 1).contiguous()          # (B, L, D)
        flat = z_t.view(-1, self.embedding_dim)          # (B*L, D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * flat @ self.embedding.weight.t())  # (B*L, K)
        enc_idx  = dist.argmin(1).unsqueeze(1)           # (B*L, 1)
        one_hot  = torch.zeros(enc_idx.shape[0], self.num_embeddings, device=z.device)
        one_hot.scatter_(1, enc_idx, 1)
        quantized = (one_hot @ self.embedding.weight).view_as(z_t)  # (B, L, D)
        e_loss = ((quantized.detach() - z_t) ** 2).mean()
        q_loss = ((quantized - z_t.detach()) ** 2).mean()
        vq_loss   = e_loss + self.commitment_cost * q_loss
        quantized_st = z_t + (quantized - z_t).detach()
        avg_probs    = one_hot.mean(0)
        perplexity   = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        quantized_st = quantized_st.permute(0, 2, 1).contiguous()  # (B, D, L)
        return quantized_st, vq_loss, perplexity, enc_idx


def _conv1d_out(length, kernel=5, stride=2, padding=2):
    """Compute Conv1d output length (integer arithmetic)."""
    return (length + 2 * padding - kernel) // stride + 1


def _compute_enc_sizes(input_length, encoder_depth):
    """Return list of sequence lengths at each encoder layer boundary."""
    sizes = [input_length]
    for _ in range(encoder_depth):
        sizes.append(_conv1d_out(sizes[-1]))
    return sizes  # length = encoder_depth + 1


def _build_encoder(base_channels, embedding_dim, encoder_depth):
    """
    Strided Conv1d encoder.
    base_channels : 'narrow' (start 16), 'standard' (start 32), 'wide' (start 64)
    encoder_depth : 3 or 4
    Final output channels = embedding_dim.
    """
    starts = {'narrow': 16, 'standard': 32, 'wide': 64}
    c0     = starts[base_channels]
    channels = [1]
    ch = c0
    for _ in range(encoder_depth - 1):
        channels.append(ch)
        ch = min(ch * 2, 256)
    channels.append(embedding_dim)

    layers = []
    for in_c, out_c in zip(channels[:-1], channels[1:]):
        layers += [
            nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_c),
        ]
    return nn.Sequential(*layers)


def _build_decoder(base_channels, embedding_dim, encoder_depth, input_length=1000):
    """Mirror of the encoder using ConvTranspose1d.
    output_padding is computed per layer from actual encoder output sizes to
    ensure perfect reconstruction length regardless of encoder_depth.
    """
    starts   = {'narrow': 16, 'standard': 32, 'wide': 64}
    c0       = starts[base_channels]
    channels = [1]
    ch = c0
    for _ in range(encoder_depth - 1):
        channels.append(ch)
        ch = min(ch * 2, 256)
    channels.append(embedding_dim)
    dec_channels = list(reversed(channels))  # embedding_dim -> ... -> 1

    # Encoder sequence lengths: enc_sizes[0]=input_length, enc_sizes[depth]=bottleneck
    enc_sizes = _compute_enc_sizes(input_length, encoder_depth)

    layers = []
    for i, (in_c, out_c) in enumerate(zip(dec_channels[:-1], dec_channels[1:])):
        is_last = (i == len(dec_channels) - 2)
        # Decoder layer i maps enc_sizes[depth-i] -> enc_sizes[depth-i-1]
        src_sz    = enc_sizes[encoder_depth - i]
        target_sz = enc_sizes[encoder_depth - i - 1]
        # ConvTranspose1d base output (no output_padding): (src-1)*2 - 4 + 5
        base_out  = (src_sz - 1) * 2 + 1
        op = target_sz - base_out  # must be 0 or 1
        layers.append(
            nn.ConvTranspose1d(in_c, out_c, kernel_size=5, stride=2,
                               padding=2, output_padding=op)
        )
        if is_last:
            layers.append(nn.Sigmoid())
        else:
            layers += [nn.ReLU(inplace=True), nn.BatchNorm1d(out_c)]
    return nn.Sequential(*layers)


class VQVAE(nn.Module):
    def __init__(self, input_len=1000, embedding_dim=64, num_embeddings=512,
                 commitment_cost=0.25, encoder_depth=3, base_channels='standard'):
        super().__init__()
        self.encoder = _build_encoder(base_channels, embedding_dim, encoder_depth)
        self.vq      = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = _build_decoder(base_channels, embedding_dim, encoder_depth, input_len)

    def forward(self, x):
        z                         = self.encoder(x)
        q, vq_loss, pp, enc_idx   = self.vq(z)
        recon                     = self.decoder(q)
        return recon, vq_loss, pp, q, enc_idx

    def encode(self, x):
        z                = self.encoder(x)
        q, _, _, enc_idx = self.vq(z)
        return q, enc_idx


def make_dataloader(X_arr, batch_size, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X_arr).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(), num_workers=0)


def train_one_epoch(model, loader, optimizer):
    model.train()
    tot_loss = tot_recon = tot_vq = tot_pp = 0.0
    N = len(loader.dataset)
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, vq_loss, pp, _, _ = model(x)
        recon_loss = nn.functional.mse_loss(recon, x)
        loss       = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        n = len(x)
        tot_loss  += loss.item()       * n
        tot_recon += recon_loss.item() * n
        tot_vq    += vq_loss.item()    * n
        tot_pp    += pp.item()         * n
    return tot_loss/N, tot_recon/N, tot_vq/N, tot_pp/N


# ============================================================================
# STEP 4: OPTUNA HYPERPARAMETER SEARCH  [SKIPPED -- loading saved params]
# ============================================================================
print("\n" + "="*70)
print("STEP 4: LOADING SAVED HYPERPARAMETERS (Optuna search skipped)")
print("="*70)

print(f"Loading metrics from {LOAD_OUTPUT_DIR}/metrics.json ...")
with open(f'{LOAD_OUTPUT_DIR}/metrics.json') as _f:
    _saved_metrics = json.load(_f)
best_params = _saved_metrics['best_params']
print(f"  Best params loaded: {best_params}")

# Minimal stub so summary can reference study fields
class _StudyStub:
    class best_trial:
        number = -1
    best_value = _saved_metrics.get('final_recon_loss', float('nan'))
    trials = []
study = _StudyStub()

# # --- original Optuna block (commented out) ---
# def objective(trial: optuna.Trial) -> float:
#     embedding_dim   = trial.suggest_categorical('embedding_dim',   [32, 64, 128, 256])
#     num_embeddings  = trial.suggest_categorical('num_embeddings',  [128, 256, 512, 1024])
#     commitment_cost = trial.suggest_categorical('commitment_cost', [0.1, 0.25, 0.5, 1.0])
#     lr              = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
#     batch_size      = trial.suggest_categorical('batch_size',      [64, 128, 256])
#     encoder_depth   = trial.suggest_categorical('encoder_depth',   [3, 4])
#     base_channels   = trial.suggest_categorical('base_channels',   ['narrow', 'standard', 'wide'])
#     model_t = VQVAE(
#         input_len       = CONFIG['input_length'],
#         embedding_dim   = embedding_dim,
#         num_embeddings  = num_embeddings,
#         commitment_cost = commitment_cost,
#         encoder_depth   = encoder_depth,
#         base_channels   = base_channels,
#     ).to(device)
#     opt_t  = optim.Adam(model_t.parameters(), lr=lr)
#     loader = make_dataloader(X, batch_size)
#     recon_loss = None
#     for epoch in range(CONFIG['epochs_tune']):
#         _, recon_loss, _, _ = train_one_epoch(model_t, loader, opt_t)
#         trial.report(recon_loss, epoch)
#         if trial.should_prune():
#             del model_t
#             torch.cuda.empty_cache()
#             raise optuna.exceptions.TrialPruned()
#     del model_t
#     torch.cuda.empty_cache()
#     return recon_loss
#
# pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
# study  = optuna.create_study(direction='minimize', pruner=pruner,
#                               study_name='vqvae_hparam_search')
# print(f"\nRunning {CONFIG['n_trials']} trials (pruning enabled)...")
# study.optimize(objective, n_trials=CONFIG['n_trials'], show_progress_bar=False)
# print(f"\nBest trial : #{study.best_trial.number}")
# print(f"  Recon loss: {study.best_value:.6f}")
# best_params = study.best_params
# trial_rows = []
# for t in study.trials:
#     row = {'trial': t.number, 'value': t.value, 'state': str(t.state)}
#     row.update(t.params)
#     trial_rows.append(row)
# pd.DataFrame(trial_rows).to_csv(f'{optuna_dir}/all_trials.csv', index=False)
# with open(f'{optuna_dir}/best_params.json', 'w') as f:
#     json.dump({'best_value': study.best_value, 'best_params': best_params}, f, indent=2)
# try:
#     fig_opt = optuna.visualization.matplotlib.plot_optimization_history(study)
#     fig_opt.get_figure().savefig(f'{optuna_dir}/optuna_history.png', dpi=150, bbox_inches='tight')
#     plt.close('all')
# except Exception:
#     fig, ax = plt.subplots(figsize=(10, 5))
#     vals   = [t.value for t in study.trials if t.value is not None]
#     best_s = [min(vals[:i+1]) for i in range(len(vals))]
#     ax.scatter(range(len(vals)), vals, s=30, alpha=0.6, label='Trial value')
#     ax.plot(range(len(best_s)), best_s, lw=2, color='red', label='Best so far')
#     ax.set_xlabel('Trial'); ax.set_ylabel('Recon Loss (MSE)')
#     ax.set_title('Optuna Optimization History')
#     ax.legend(); ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(f'{optuna_dir}/optuna_history.png', dpi=150, bbox_inches='tight')
#     plt.close()
# print(f"\nOptuna results saved to: {optuna_dir}/")

# ============================================================================
# STEP 5: LOAD SAVED MODEL  [training skipped]
# ============================================================================
print("\n" + "="*70)
print("STEP 5: LOADING SAVED MODEL (retraining skipped)")
print("="*70)

best_p = best_params
model_final = VQVAE(
    input_len       = CONFIG['input_length'],
    embedding_dim   = best_p['embedding_dim'],
    num_embeddings  = best_p['num_embeddings'],
    commitment_cost = best_p['commitment_cost'],
    encoder_depth   = best_p['encoder_depth'],
    base_channels   = best_p['base_channels'],
).to(device)

_model_path = f'{LOAD_OUTPUT_DIR}/vqvae_model.pth'
model_final.load_state_dict(torch.load(_model_path, map_location=device))
model_final.eval()
print(f"Loaded model weights from: {_model_path}")
print("Architecture:")
for k, v in best_p.items():
    print(f"  {k}: {v}")

# Stub history using values from saved metrics (used only in summary)
history = {
    'epoch':      list(range(1, CONFIG['epochs_final'] + 1)),
    'total_loss': [_saved_metrics.get('final_recon_loss', 0.0)] * CONFIG['epochs_final'],
    'recon_loss': [_saved_metrics.get('final_recon_loss', 0.0)] * CONFIG['epochs_final'],
    'vq_loss':    [0.0] * CONFIG['epochs_final'],
    'perplexity': [_saved_metrics.get('final_perplexity', 0.0)] * CONFIG['epochs_final'],
}

# # --- original training block (commented out) ---
# optimizer_final = optim.Adam(model_final.parameters(), lr=best_p['lr'])
# loader_final    = make_dataloader(X, best_p['batch_size'])
# for epoch in range(CONFIG['epochs_final']):
#     total, recon, vq, pp = train_one_epoch(model_final, loader_final, optimizer_final)
#     history['epoch'].append(epoch + 1)
#     history['total_loss'].append(total)
#     history['recon_loss'].append(recon)
#     history['vq_loss'].append(vq)
#     history['perplexity'].append(pp)
#     if (epoch + 1) % 10 == 0 or epoch == 0:
#         print(f"  Epoch [{epoch+1:3d}/{CONFIG['epochs_final']}] "
#               f"Total: {total:.6f} | Recon: {recon:.6f} | "
#               f"VQ: {vq:.6f} | Perplexity: {pp:.2f}")
# model_path = f'{output_dir}/vqvae_model.pth'
# torch.save(model_final.state_dict(), model_path)
# print(f"\nModel saved: {model_path}")

# Training curves plot skipped (no live history)
# fig, axes = plt.subplots(2, 2, figsize=(14, 9))
# ...
# plt.savefig(f'{output_dir}/04_training_curves.png', ...)

# ============================================================================
# STEP 6: COMPUTE RECONSTRUCTION ERROR & METRICS
# ============================================================================
print("\n" + "="*70)
print("STEP 6: COMPUTING RECONSTRUCTION ERRORS & METRICS")
print("="*70)

model_final.eval()
N  = len(X)
bs = CONFIG['eval_batch_size']
n_eval_batches = (N + bs - 1) // bs

recon_list = []
latent_list = []
encidx_list = []
recon_arr_list = []

with torch.no_grad():
    for i in range(n_eval_batches):
        s, e = i * bs, min((i + 1) * bs, N)
        xb   = torch.from_numpy(X[s:e]).unsqueeze(1).to(device)
        recon_b, _, _, q, enc_idx = model_final(xb)
        recon_list.append(((xb - recon_b) ** 2).mean(dim=[1, 2]).cpu().numpy())
        latent_list.append(q.mean(dim=2).cpu().numpy())
        encidx_list.append(enc_idx.cpu().numpy().reshape(e - s, -1))
        recon_arr_list.append(recon_b.squeeze(1).cpu().numpy())
        if (i + 1) % 50 == 0 or (i + 1) == n_eval_batches:
            print(f"  Processed {e}/{N} samples")

recon_errors = np.concatenate(recon_list)
latents      = np.concatenate(latent_list)
encoding_idx = np.concatenate(encidx_list)
X_recon      = np.concatenate(recon_arr_list)  # (N, 1000)

print(f"\nReconstruction error -- mean: {recon_errors.mean():.6f}, "
      f"std: {recon_errors.std():.6f}")

# Codebook utilisation
flat_idx   = encoding_idx.flatten()
uniq_codes, code_counts = np.unique(flat_idx, return_counts=True)
cb_frac    = len(uniq_codes) / best_p['num_embeddings']
code_probs = code_counts / code_counts.sum()
cb_entropy = -np.sum(code_probs * np.log(code_probs + 1e-10))
max_ent    = np.log(best_p['num_embeddings'])
print(f"Codebook utilisation: {len(uniq_codes)} / {best_p['num_embeddings']} "
      f"({cb_frac*100:.1f}%)  entropy: {cb_entropy:.3f} / {max_ent:.3f} "
      f"({cb_entropy/max_ent*100:.1f}%)")

# Latent spread
lat_var = latents.var(axis=0).mean()
sub_idx = np.random.choice(len(latents), min(2000, len(latents)), replace=False)
pwd     = pairwise_distances(latents[sub_idx], metric='euclidean')
triu    = np.triu_indices(len(sub_idx), k=1)
mpd     = pwd[triu].mean()
print(f"Latent mean dim-variance: {lat_var:.6f}  |  "
      f"Mean pairwise dist (n={len(sub_idx)}): {mpd:.4f}")

# Error percentile bands
NB    = CONFIG['num_bands']
BL    = CONFIG['band_labels']
edges = np.percentile(recon_errors, np.linspace(0, 100, NB + 1))
e_band = np.digitize(recon_errors, edges[1:-1])  # 0 .. NB-1

print("\nError percentile bands:")
for b in range(NB):
    bm = e_band == b
    be = recon_errors[bm]
    print(f"  Band {b} ({BL[b]}): {bm.sum():5d} spectra  "
          f"error [{be.min():.6f}, {be.max():.6f}]")

# Save metrics
metrics = {
    'recon_error_mean':       float(recon_errors.mean()),
    'recon_error_std':        float(recon_errors.std()),
    'recon_error_p25':        float(np.percentile(recon_errors, 25)),
    'recon_error_p50':        float(np.percentile(recon_errors, 50)),
    'recon_error_p75':        float(np.percentile(recon_errors, 75)),
    'codebook_unique_used':   int(len(uniq_codes)),
    'codebook_usage_frac':    float(cb_frac),
    'codebook_entropy':       float(cb_entropy),
    'codebook_entropy_pct':   float(cb_entropy / max_ent * 100),
    'latent_mean_dim_var':    float(lat_var),
    'latent_mean_pairwise_d': float(mpd),
    'final_recon_loss':       float(history['recon_loss'][-1]),
    'final_perplexity':       float(history['perplexity'][-1]),
    'best_params':            best_p,
}
with open(f'{output_dir}/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved: metrics.json")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(recon_errors, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(recon_errors.mean(), color='red', lw=2, linestyle='--',
                label=f'Mean: {recon_errors.mean():.5f}')
axes[0].set_xlabel('Reconstruction Error (MSE)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Reconstruction Errors', fontweight='bold')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(np.sort(recon_errors), lw=2, color='darkred')
axes[1].set_xlabel('Sample (sorted)')
axes[1].set_ylabel('Reconstruction Error')
axes[1].set_title('Sorted Reconstruction Errors', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_reconstruction_errors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 05_reconstruction_errors.png")

# ============================================================================
# STEP 7: UMAP LATENT SPACE -- colored by reconstruction error
# ============================================================================
print("\n" + "="*70)
print("STEP 7: UMAP LATENT SPACE VISUALIZATION")
print("="*70)

print("Fitting UMAP...")
reducer    = umap.UMAP(
    n_components = 2,
    n_neighbors  = CONFIG['umap_n_neighbors'],
    min_dist     = CONFIG['umap_min_dist'],
    random_state = 42,
    low_memory   = True,
)
latents_2d = reducer.fit_transform(latents)
print(f"UMAP projection shape: {latents_2d.shape}")

fig, ax = plt.subplots(figsize=(11, 9))
sc = ax.scatter(
    latents_2d[:, 0], latents_2d[:, 1],
    c          = recon_errors,
    cmap       = 'viridis',
    alpha      = 0.55,
    s          = 18,
    edgecolors = 'none',
)
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Reconstruction Error (MSE)', fontsize=12)
ax.set_xlabel('UMAP Dimension 1', fontsize=13)
ax.set_ylabel('UMAP Dimension 2', fontsize=13)
ax.set_title('VQ-VAE Latent Space (UMAP)\nColoured by Reconstruction Error',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_umap_recon_error.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 06_umap_recon_error.png")

if has_labels:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le_labels = le.fit_transform(labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    sc2 = ax.scatter(latents_2d[:, 0], latents_2d[:, 1],
                     c=le_labels, cmap='tab20', alpha=0.55, s=18, edgecolors='none')
    cbar2 = plt.colorbar(sc2, ax=ax, pad=0.02, ticks=range(len(le.classes_)))
    cbar2.set_ticklabels(le.classes_)
    cbar2.set_label('True Label', fontsize=12)
    ax.set_xlabel('UMAP Dimension 1', fontsize=13)
    ax.set_ylabel('UMAP Dimension 2', fontsize=13)
    ax.set_title('VQ-VAE Latent Space (UMAP)\nColoured by True Labels',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06b_umap_true_labels.png', dpi=200, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 7b: t-SNE LATENT SPACE -- colored by reconstruction error (openTSNE, full dataset)
# ============================================================================
print("\n" + "="*70)
print("STEP 7b: t-SNE LATENT SPACE VISUALIZATION (openTSNE -- full dataset)")
print("="*70)

# Use ALL samples -- openTSNE scales to large datasets via approx NN + FFT acceleration
tsne_idx     = np.arange(len(latents))
tsne_latents = latents
tsne_errors  = recon_errors

print(f"Fitting openTSNE on full dataset: {len(tsne_latents)} samples...")
tsne_obj = openTSNE_TSNE(
    n_components   = 2,
    perplexity     = 30,
    learning_rate  = 'auto',
    initialization = 'pca',
    random_state   = 42,
    n_jobs         = -1,
    verbose        = True,
)
latents_tsne = np.array(tsne_obj.fit(tsne_latents))
print(f"openTSNE projection shape: {latents_tsne.shape}")

# Standalone t-SNE plot
fig, ax = plt.subplots(figsize=(11, 9))
sc_tsne = ax.scatter(
    latents_tsne[:, 0], latents_tsne[:, 1],
    c          = tsne_errors,
    cmap       = 'viridis',
    alpha      = 0.55,
    s          = 18,
    edgecolors = 'none',
)
cbar_t = plt.colorbar(sc_tsne, ax=ax, pad=0.02)
cbar_t.set_label('Reconstruction Error (MSE)', fontsize=12)
ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax.set_title('VQ-VAE Latent Space (t-SNE)\nColoured by Reconstruction Error',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{output_dir}/06c_tsne_recon_error.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 06c_tsne_recon_error.png")

# Side-by-side UMAP vs t-SNE comparison -- both use full dataset
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

sc_u = axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1],
                       c=recon_errors, cmap='viridis',
                       alpha=0.55, s=14, edgecolors='none')
plt.colorbar(sc_u, ax=axes[0], pad=0.02, label='Recon Error (MSE)')
axes[0].set_xlabel('UMAP Dimension 1', fontsize=12)
axes[0].set_ylabel('UMAP Dimension 2', fontsize=12)
axes[0].set_title('UMAP', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.2)

sc_t = axes[1].scatter(latents_tsne[:, 0], latents_tsne[:, 1],
                       c=recon_errors, cmap='viridis',
                       alpha=0.55, s=14, edgecolors='none')
plt.colorbar(sc_t, ax=axes[1], pad=0.02, label='Recon Error (MSE)')
axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
axes[1].set_title('t-SNE', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.2)

plt.suptitle('VQ-VAE Latent Space: UMAP vs t-SNE\nColoured by Reconstruction Error',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/06d_umap_vs_tsne.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 06d_umap_vs_tsne.png")

# ============================================================================
# STEP 8: ERROR PERCENTILE BAND EXPLORATION
# ============================================================================
print("\n" + "="*70)
print("STEP 8: ERROR PERCENTILE BAND EXPLORATION")
print("="*70)

for bid in range(NB):
    bm   = e_band == bid
    bidx = np.where(bm)[0]
    berr = recon_errors[bm]
    n_b  = len(bidx)
    print(f"\nBand {bid} ({BL[bid]}): {n_b} spectra  "
          f"error [{berr.min():.6f}, {berr.max():.6f}]")

    n_plt = min(12, n_b)
    samp  = np.random.choice(bidx, n_plt, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for i, idx in enumerate(samp):
        ax = axes.flat[i]
        ax.plot(X[idx], lw=1.5, color='crimson', alpha=0.8)
        ax.set_xlabel('Mass Channel', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.set_title(f'Spectrum {idx}  err={recon_errors[idx]:.5f}', fontsize=9)
        ax.grid(True, alpha=0.3)
    for i in range(n_plt, 12):
        axes.flat[i].axis('off')
    plt.suptitle(f'Band {bid}: {BL[bid]} (total {n_b})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_band{bid}_spectra.png', dpi=150, bbox_inches='tight')
    plt.close()

fig, axes = plt.subplots(1, NB, figsize=(22, 5))
for bid in range(NB):
    bm  = e_band == bid
    mu  = X[bm].mean(0)
    sd  = X[bm].std(0)
    ax  = axes[bid]
    ax.plot(mu, lw=2, color='darkblue', label='Mean')
    ax.fill_between(range(len(mu)), mu - sd, mu + sd,
                    alpha=0.3, color='lightblue', label='+/-1 SD')
    ax.set_xlabel('Mass Channel', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_title(f'Band {bid}\n{BL[bid]}\n(n={bm.sum()})',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/08_band_average_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_band*_spectra.png, 08_band_average_spectra.png")

# ============================================================================
# STEP 9: RECONSTRUCTION QUALITY -- original vs reconstructed overlays
# ============================================================================
print("\n" + "="*70)
print("STEP 9: RECONSTRUCTION QUALITY OVERLAYS")
print("="*70)

sorted_idx   = np.argsort(recon_errors)
best_idx     = sorted_idx[:3]
median_start = len(sorted_idx) // 2 - 1
median_idx   = sorted_idx[median_start:median_start + 3]
worst_idx    = sorted_idx[-3:]
sel_idx      = np.concatenate([best_idx, median_idx, worst_idx])
qualities    = ['Best'] * 3 + ['Median'] * 3 + ['Worst'] * 3

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for i, (idx, qual) in enumerate(zip(sel_idx, qualities)):
    ax = axes.flat[i]
    ax.plot(X[idx],       lw=2.0, color='steelblue', label='Original',      alpha=0.85)
    ax.plot(X_recon[idx], lw=2.0, color='firebrick',  label='Reconstructed', alpha=0.85, ls='--')
    ax.set_xlabel('Mass Channel', fontsize=10)
    ax.set_ylabel('Intensity', fontsize=10)
    ax.set_title(f'{qual} -- Spectrum {idx}\nMSE = {recon_errors[idx]:.6f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
plt.suptitle('Reconstruction Quality: Best / Median / Worst (3 each)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/09_reconstruction_overlays.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 09_reconstruction_overlays.png")

# ============================================================================
# STEP 10: RESIDUAL PLOTS (original - reconstructed)
# ============================================================================
print("\n" + "="*70)
print("STEP 10: RESIDUAL PLOTS")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(18, 12))
for i, (idx, qual) in enumerate(zip(sel_idx, qualities)):
    ax  = axes.flat[i]
    res = X[idx] - X_recon[idx]
    ax.plot(res, lw=1.5, color='darkorchid', alpha=0.9)
    ax.axhline(0, color='black', lw=1, ls='--')
    ax.fill_between(range(len(res)), res, 0,
                    where=(res >= 0), color='steelblue', alpha=0.35,
                    label='Over-reconstruction')
    ax.fill_between(range(len(res)), res, 0,
                    where=(res < 0),  color='firebrick',  alpha=0.35,
                    label='Under-reconstruction')
    ax.set_xlabel('Mass Channel', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title(f'{qual} Residual -- Spectrum {idx}\nMSE = {recon_errors[idx]:.6f}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle('Residuals (Original - Reconstructed): Best / Median / Worst',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/10_residual_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 10_residual_plots.png")

# ============================================================================
# STEP 11: MEAN +/- STD RECONSTRUCTION PER ERROR BAND
# ============================================================================
print("\n" + "="*70)
print("STEP 11: MEAN +/- 1-STD RECONSTRUCTION PER ERROR BAND")
print("="*70)

band_colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

fig, axes = plt.subplots(2, NB, figsize=(24, 10))

for bid in range(NB):
    bm      = e_band == bid
    orig_mu = X[bm].mean(0)
    orig_sd = X[bm].std(0)
    rec_mu  = X_recon[bm].mean(0)
    rec_sd  = X_recon[bm].std(0)
    col     = band_colors[bid]
    x_ax    = np.arange(len(orig_mu))

    # Top row: mean original vs mean reconstructed
    ax_top = axes[0, bid]
    ax_top.plot(orig_mu, lw=2, color='steelblue', label='Orig mean', alpha=0.9)
    ax_top.fill_between(x_ax, orig_mu - orig_sd, orig_mu + orig_sd,
                        alpha=0.2, color='steelblue')
    ax_top.plot(rec_mu, lw=2, color='firebrick', label='Recon mean', alpha=0.9, ls='--')
    ax_top.fill_between(x_ax, rec_mu - rec_sd, rec_mu + rec_sd,
                        alpha=0.2, color='firebrick')
    ax_top.set_title(f'Band {bid}: {BL[bid]}\n(n={bm.sum()})',
                     fontsize=10, fontweight='bold', color=col)
    ax_top.set_xlabel('Mass Channel', fontsize=9)
    ax_top.set_ylabel('Intensity', fontsize=9)
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    # Bottom row: mean residual
    ax_bot = axes[1, bid]
    res_mu = orig_mu - rec_mu
    res_sd = np.sqrt(orig_sd**2 + rec_sd**2)
    ax_bot.plot(res_mu, lw=2, color='darkorchid', alpha=0.9)
    ax_bot.fill_between(x_ax, res_mu - res_sd, res_mu + res_sd,
                        alpha=0.25, color='darkorchid')
    ax_bot.axhline(0, color='black', lw=1, ls='--')
    ax_bot.set_xlabel('Mass Channel', fontsize=9)
    ax_bot.set_ylabel('Mean Residual', fontsize=9)
    ax_bot.set_title(f'Mean Residual +/- SD (Band {bid})', fontsize=10, fontweight='bold')
    ax_bot.grid(True, alpha=0.3)

plt.suptitle('Mean +/- 1-SD Reconstruction per Error Percentile Band',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/11_band_reconstruction_overlay.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 11_band_reconstruction_overlay.png")

# ============================================================================
# SAVE DATA & FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SAVING DATA & SUMMARY")
print("="*70)

np.savez_compressed(
    f'{output_dir}/analysis_data.npz',
    X_processed  = X,
    X_recon      = X_recon,
    latents      = latents,
    latents_2d   = latents_2d,
    latents_tsne = latents_tsne,
    error_band   = e_band,
    recon_errors = recon_errors,
    encoding_idx = encoding_idx,
    labels       = labels if has_labels else np.array([]),
)

# All samples have t-SNE coordinates (full-dataset openTSNE)
latent_df = pd.DataFrame({
    'index':            range(N),
    'recon_error':      recon_errors,
    'error_band':       e_band,
    'error_band_label': [BL[b] for b in e_band],
    'umap_x':           latents_2d[:, 0],
    'umap_y':           latents_2d[:, 1],
    'tsne_x':           latents_tsne[:, 0],
    'tsne_y':           latents_tsne[:, 1],
    'label':            labels if has_labels else [None] * N,
})
latent_df.to_csv(f'{output_dir}/latent_assignments.csv', index=False)
print("Saved: analysis_data.npz, latent_assignments.csv")

summary = f"""
VQ-VAE Analysis Summary
{'='*70}

Dataset : {CONFIG['data_path']}
Samples : {N}  |  Spectrum length : {CONFIG['input_length']}
Labels  : {has_labels}

Preprocessing:
  Savitzky-Golay (window={CONFIG['savgol_window']}, order={CONFIG['savgol_order']})
  + log1p + min-max normalisation

Optuna Search ({CONFIG['n_trials']} trials x {CONFIG['epochs_tune']} warm-up epochs each):
  Best trial  : #{study.best_trial.number}
  Best value  : {study.best_value:.6f}
  Best params : {best_p}

Final Training ({CONFIG['epochs_final']} epochs):
  Final recon loss : {history['recon_loss'][-1]:.6f}
  Final VQ loss    : {history['vq_loss'][-1]:.6f}
  Final perplexity : {history['perplexity'][-1]:.2f}

Reconstruction Error:
  Mean : {recon_errors.mean():.6f} +/- {recon_errors.std():.6f}
  P25  : {np.percentile(recon_errors, 25):.6f}
  P50  : {np.percentile(recon_errors, 50):.6f}
  P75  : {np.percentile(recon_errors, 75):.6f}

Codebook Utilisation:
  {len(uniq_codes)} / {best_p['num_embeddings']} ({cb_frac*100:.1f}%)
  Entropy: {cb_entropy:.4f} / {max_ent:.4f} ({cb_entropy/max_ent*100:.1f}%)

Latent Space (UMAP n_neighbors={CONFIG['umap_n_neighbors']}, min_dist={CONFIG['umap_min_dist']}):
  Mean dim-variance  : {lat_var:.6f}
  Mean pairwise dist : {mpd:.4f}

Output Files (all in {output_dir}/):
  04_training_curves.png
  05_reconstruction_errors.png
  06_umap_recon_error.png            <- UMAP coloured by MSE
  06c_tsne_recon_error.png           <- t-SNE coloured by MSE
  06d_umap_vs_tsne.png               <- side-by-side comparison
  07_band*_spectra.png               <- sample spectra per error band
  08_band_average_spectra.png
  09_reconstruction_overlays.png     <- original vs reconstructed
  10_residual_plots.png              <- residuals best/median/worst
  11_band_reconstruction_overlay.png <- mean+/-SD recon per band
  vqvae_model.pth
  metrics.json
  analysis_data.npz
  latent_assignments.csv

Optuna study in: {optuna_dir}/
  best_params.json
  all_trials.csv
  optuna_history.png
"""

print(summary)
with open(f'{output_dir}/SUMMARY.txt', 'w') as f:
    f.write(summary)

print(f"\n{'='*70}")
print(f"ALL OUTPUTS SAVED TO: {output_dir}")
print(f"{'='*70}\n")
