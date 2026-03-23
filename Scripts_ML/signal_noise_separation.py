"""
Signal vs Noise Separation via Signal-Only VQ-VAE Retraining

Strategy:
  1. Load outputs from the prior full-dataset VQ-VAE run
  2. Extract band-0 spectra (bottom 20% reconstruction error) as the
     signal-only training set
  3. Build a new VQ-VAE with a tiny codebook (16 codes) and warm-start
     the encoder/decoder from the existing checkpoint
  4. Train with peak-weighted MSE loss so flat noise baselines contribute
     minimally — the codebook learns only signal prototypes
  5. Re-evaluate all 69 938 spectra with the signal-only model
     → noise fails to map onto the signal codebook → high error
  6. Otsu threshold on the re-computed error histogram → binary labels
  7. Save outputs to outputs/signal_noise/signal_noise_<timestamp>/

Inputs (set PREV_RUN_DIR and PREV_MODEL_DIR below):
  PREV_RUN_DIR/analysis_data.npz      — X_processed, latents_tsne, error_band, recon_errors
  PREV_MODEL_DIR/vqvae_model.pth      — encoder/decoder weights to warm-start from
"""

# ============================================================================
# Imports
# ============================================================================
import subprocess, sys

# Ensure scikit-image is available for Otsu
try:
    from skimage.filters import threshold_otsu
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image", "--quiet"])
    from skimage.filters import threshold_otsu

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

# ============================================================================
# Configuration  ── edit these two paths before running
# ============================================================================
# Directory that contains analysis_data.npz and latent_assignments.csv
PREV_RUN_DIR   = 'outputs/unknown_vqvae/unknown_vqvae_analysis_latest'

# Directory that contains the saved vqvae_model.pth
PREV_MODEL_DIR = 'outputs/unknown_vqvae/unknown_vqvae_analysis_20260310_211416'

CONFIG = {
    # Architecture (must match the checkpoint to allow weight transfer)
    'embedding_dim':      64,
    'encoder_depth':      3,
    'base_channels':      'standard',
    # NEW: tiny codebook — only enough capacity for signal prototypes
    'num_embeddings_new': 16,
    'commitment_cost':    1.0,
    'input_length':       1000,
    # Peak-weighted loss: weight = 1 + scale * x  (x in [0,1])
    # scale=9 → 10× weight at peak tops, 1× at zero baseline
    'peak_weight_scale':  9.0,
    # Training
    'epochs':             100,
    'batch_size':         64,
    'lr':                 1e-3,
    'eval_batch_size':    256,
}

# ============================================================================
# Setup
# ============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'outputs/signal_noise/signal_noise_{timestamp}'
os.makedirs(output_dir, exist_ok=True)
print(f"\nOutput directory: {output_dir}")

with open(f'{output_dir}/config.json', 'w') as f:
    json.dump({**CONFIG, 'PREV_RUN_DIR': PREV_RUN_DIR,
               'PREV_MODEL_DIR': PREV_MODEL_DIR}, f, indent=2)

# ============================================================================
# STEP 1: Load previous run outputs
# ============================================================================
print("\n" + "="*70)
print("STEP 1: LOADING PREVIOUS RUN DATA")
print("="*70)

npz_path = f'{PREV_RUN_DIR}/analysis_data.npz'
print(f"Loading {npz_path} ...")
npz = np.load(npz_path, allow_pickle=True)

X_processed  = npz['X_processed'].astype(np.float32)   # (N, 1000)
error_band   = npz['error_band']                         # (N,)  int 0–4
recon_errors = npz['recon_errors']                       # (N,)  float MSE
latents_tsne = npz['latents_tsne']                       # (N, 2) openTSNE coords

N = len(X_processed)
print(f"  Total spectra     : {N}")
print(f"  Spectrum length   : {X_processed.shape[1]}")
print(f"  error_band range  : {error_band.min()} – {error_band.max()}")
print(f"  recon_error range : {recon_errors.min():.6f} – {recon_errors.max():.6f}")

# ============================================================================
# STEP 2: Extract band-0 signal training set
# ============================================================================
print("\n" + "="*70)
print("STEP 2: EXTRACTING BAND-0 SIGNAL TRAINING SET")
print("="*70)

signal_mask = error_band == 0
X_signal    = X_processed[signal_mask]
signal_idx  = np.where(signal_mask)[0]
signal_errs = recon_errors[signal_mask]

n_signal = len(X_signal)
print(f"  Band-0 spectra (bottom 20% MSE): {n_signal} / {N} "
      f"({n_signal / N * 100:.1f}%)")
print(f"  Band-0 MSE range : [{signal_errs.min():.6f}, {signal_errs.max():.6f}]")
print(f"  Band-0 MSE mean  : {signal_errs.mean():.6f} ± {signal_errs.std():.6f}")

# Visual check: 20 random band-0 spectra
n_vis   = min(20, n_signal)
vis_idx = np.random.choice(n_signal, n_vis, replace=False)
fig, axes = plt.subplots(4, 5, figsize=(20, 14))
for i, vi in enumerate(vis_idx):
    ax = axes.flat[i]
    ax.plot(X_signal[vi], lw=1.5, color='steelblue', alpha=0.85)
    ax.set_title(f'Band-0 #{signal_idx[vi]}\nMSE={signal_errs[vi]:.5f}', fontsize=8)
    ax.set_xlabel('Mass Channel', fontsize=8)
    ax.set_ylabel('Intensity', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'Band-0 Signal Training Samples (n={n_signal})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/01_band0_sample_spectra.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_band0_sample_spectra.png")

# ============================================================================
# VQ-VAE ARCHITECTURE  (identical helpers to the original analysis script)
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
        z_t  = z.permute(0, 2, 1).contiguous()           # (B, L, D)
        flat = z_t.view(-1, self.embedding_dim)           # (B*L, D)
        dist = (flat.pow(2).sum(1, keepdim=True)
                + self.embedding.weight.pow(2).sum(1)
                - 2 * flat @ self.embedding.weight.t())   # (B*L, K)
        # Per-position min distance to nearest code (raw VQ "surprise" score)
        min_dist, _ = dist.min(dim=1)                     # (B*L,)
        enc_idx  = dist.argmin(1).unsqueeze(1)
        one_hot  = torch.zeros(enc_idx.shape[0], self.num_embeddings, device=z.device)
        one_hot.scatter_(1, enc_idx, 1)
        quantized    = (one_hot @ self.embedding.weight).view_as(z_t)
        e_loss       = ((quantized.detach() - z_t) ** 2).mean()
        q_loss       = ((quantized - z_t.detach()) ** 2).mean()
        vq_loss      = e_loss + self.commitment_cost * q_loss
        quantized_st = z_t + (quantized - z_t).detach()
        avg_probs    = one_hot.mean(0)
        perplexity   = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        quantized_st = quantized_st.permute(0, 2, 1).contiguous()
        # Mean per-sample min-dist (shape: B)
        mean_min_dist = min_dist.view(z.shape[0], -1).mean(dim=1)
        return quantized_st, vq_loss, perplexity, enc_idx, mean_min_dist


def _conv1d_out(length, kernel=5, stride=2, padding=2):
    return (length + 2 * padding - kernel) // stride + 1


def _compute_enc_sizes(input_length, encoder_depth):
    sizes = [input_length]
    for _ in range(encoder_depth):
        sizes.append(_conv1d_out(sizes[-1]))
    return sizes


def _build_encoder(base_channels, embedding_dim, encoder_depth):
    starts   = {'narrow': 16, 'standard': 32, 'wide': 64}
    c0       = starts[base_channels]
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
    starts   = {'narrow': 16, 'standard': 32, 'wide': 64}
    c0       = starts[base_channels]
    channels = [1]
    ch = c0
    for _ in range(encoder_depth - 1):
        channels.append(ch)
        ch = min(ch * 2, 256)
    channels.append(embedding_dim)
    dec_channels = list(reversed(channels))
    enc_sizes    = _compute_enc_sizes(input_length, encoder_depth)
    layers = []
    for i, (in_c, out_c) in enumerate(zip(dec_channels[:-1], dec_channels[1:])):
        is_last  = (i == len(dec_channels) - 2)
        src_sz   = enc_sizes[encoder_depth - i]
        tgt_sz   = enc_sizes[encoder_depth - i - 1]
        base_out = (src_sz - 1) * 2 + 1
        op       = tgt_sz - base_out
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
    def __init__(self, input_len=1000, embedding_dim=64, num_embeddings=16,
                 commitment_cost=1.0, encoder_depth=3, base_channels='standard'):
        super().__init__()
        self.encoder = _build_encoder(base_channels, embedding_dim, encoder_depth)
        self.vq      = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = _build_decoder(base_channels, embedding_dim, encoder_depth, input_len)

    def forward(self, x):
        z                                  = self.encoder(x)
        q, vq_loss, pp, enc_idx, min_dist  = self.vq(z)
        recon                              = self.decoder(q)
        return recon, vq_loss, pp, q, enc_idx, min_dist

    def encode(self, x):
        z                           = self.encoder(x)
        q, _, _, enc_idx, min_dist  = self.vq(z)
        return q, enc_idx, min_dist


# ============================================================================
# Peak-weighted MSE loss
# ============================================================================

def peak_weighted_mse(recon, target, scale=9.0):
    """
    weight = 1 + scale * target   (target in [0, 1])
    At baseline (target=0): weight = 1    (normal MSE contribution)
    At peak top  (target=1): weight = 10  (10× with default scale=9)
    Forces the model to prioritise reconstructing peaks faithfully over
    flat baseline regions that noise spectra share.
    """
    weight = 1.0 + scale * target
    return (weight * (recon - target).pow(2)).mean()


# ============================================================================
# STEP 3: Build model and warm-start from checkpoint
# ============================================================================
print("\n" + "="*70)
print("STEP 3: BUILDING SIGNAL-ONLY VQ-VAE + WARM START")
print("="*70)

model = VQVAE(
    input_len       = CONFIG['input_length'],
    embedding_dim   = CONFIG['embedding_dim'],
    num_embeddings  = CONFIG['num_embeddings_new'],
    commitment_cost = CONFIG['commitment_cost'],
    encoder_depth   = CONFIG['encoder_depth'],
    base_channels   = CONFIG['base_channels'],
).to(device)

ckpt_path = f'{PREV_MODEL_DIR}/vqvae_model.pth'
print(f"Loading checkpoint: {ckpt_path}")
ckpt     = torch.load(ckpt_path, map_location='cpu')
model_sd = model.state_dict()

loaded, skipped = [], []
for k, v in ckpt.items():
    if k.startswith('vq.'):
        skipped.append(k)           # VQ layer: reinitialise fresh for signal-only
    elif k in model_sd and model_sd[k].shape == v.shape:
        model_sd[k] = v
        loaded.append(k)
    else:
        skipped.append(k)
model.load_state_dict(model_sd)

print(f"  Warm-started layers : {len(loaded)}")
print(f"  Skipped / fresh     : {len(skipped)}  (VQ layer reinitialised)")
print(f"  Codebook size       : {CONFIG['num_embeddings_new']} codes (down from 128)")
print(f"  Peak weight scale   : {CONFIG['peak_weight_scale']}  (10× at peak tops)")

# ============================================================================
# STEP 4: Train on signal-only subset  (band-0 spectra, 100 epochs)
# ============================================================================
print("\n" + "="*70)
print("STEP 4: TRAINING ON SIGNAL-ONLY SUBSET (band 0)")
print("="*70)
print(f"  Training samples : {n_signal}")
print(f"  Epochs           : {CONFIG['epochs']}")
print(f"  Batch size       : {CONFIG['batch_size']}")
print(f"  LR               : {CONFIG['lr']}")

ds_signal = TensorDataset(torch.from_numpy(X_signal).unsqueeze(1))
loader    = DataLoader(ds_signal, batch_size=CONFIG['batch_size'], shuffle=True,
                       pin_memory=torch.cuda.is_available(), num_workers=0)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG['epochs'], eta_min=1e-5)

history = {'epoch': [], 'total': [], 'recon': [], 'vq': [], 'perplexity': []}
scale   = CONFIG['peak_weight_scale']

for epoch in range(1, CONFIG['epochs'] + 1):
    model.train()
    tot_total = tot_recon = tot_vq = tot_pp = 0.0
    Nb = len(ds_signal)
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon, vq_loss, pp, _, _, _ = model(x)
        recon_loss = peak_weighted_mse(recon, x, scale=scale)
        loss       = recon_loss + vq_loss
        loss.backward()
        optimizer.step()
        n = len(x)
        tot_total += loss.item()       * n
        tot_recon += recon_loss.item() * n
        tot_vq    += vq_loss.item()    * n
        tot_pp    += pp.item()         * n
    scheduler.step()

    history['epoch'].append(epoch)
    history['total'].append(tot_total / Nb)
    history['recon'].append(tot_recon / Nb)
    history['vq'].append(tot_vq    / Nb)
    history['perplexity'].append(tot_pp    / Nb)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch [{epoch:3d}/{CONFIG['epochs']}]  "
              f"Total: {history['total'][-1]:.6f}  "
              f"Recon: {history['recon'][-1]:.6f}  "
              f"VQ: {history['vq'][-1]:.6f}  "
              f"Perplexity: {history['perplexity'][-1]:.2f}")

model_path = f'{output_dir}/signal_only_model.pth'
torch.save(model.state_dict(), model_path)
print(f"\nModel saved: {model_path}")

# Training curves
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ep = history['epoch']
axes[0, 0].plot(ep, history['total'],      lw=2, color='steelblue')
axes[0, 0].set_title('Total Loss',               fontweight='bold')
axes[0, 1].plot(ep, history['recon'],      lw=2, color='darkgreen')
axes[0, 1].set_title('Peak-Weighted Recon Loss',  fontweight='bold')
axes[1, 0].plot(ep, history['vq'],         lw=2, color='firebrick')
axes[1, 0].set_title('VQ Loss',                   fontweight='bold')
axes[1, 1].plot(ep, history['perplexity'], lw=2, color='darkorchid')
axes[1, 1].set_title('Codebook Perplexity',        fontweight='bold')
for ax in axes.flat:
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
plt.suptitle('Signal-Only VQ-VAE Training Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_training_curves.png")

# ============================================================================
# STEP 5: Evaluate ALL spectra with the signal-only model
# ============================================================================
print("\n" + "="*70)
print("STEP 5: EVALUATING ALL SPECTRA WITH SIGNAL-ONLY MODEL")
print("="*70)

model.eval()
bs  = CONFIG['eval_batch_size']
n_b = (N + bs - 1) // bs

recon_v2_list  = []
vqdist_list    = []
recon_arr_list = []
latent_v2_list = []   # for silhouette score

with torch.no_grad():
    for i in range(n_b):
        s, e = i * bs, min((i + 1) * bs, N)
        xb   = torch.from_numpy(X_processed[s:e]).unsqueeze(1).to(device)
        recon_b, _, _, q_b, _, min_d = model(xb)
        # Peak-weighted per-sample MSE
        w_b   = 1.0 + scale * xb
        err_b = (w_b * (recon_b - xb).pow(2)).mean(dim=[1, 2])
        recon_v2_list.append(err_b.cpu().numpy())
        vqdist_list.append(min_d.cpu().numpy())
        recon_arr_list.append(recon_b.squeeze(1).cpu().numpy())
        # Mean-pool quantised latent over time dimension → (batch, embedding_dim)
        latent_v2_list.append(q_b.mean(dim=-1).cpu().numpy())
        if (i + 1) % 50 == 0 or (i + 1) == n_b:
            print(f"  Evaluated {e}/{N}")

recon_error_v2 = np.concatenate(recon_v2_list)
vq_min_dist    = np.concatenate(vqdist_list)
X_recon_v2     = np.concatenate(recon_arr_list)
new_latents    = np.concatenate(latent_v2_list)   # (N, embedding_dim)

print(f"\nNew recon error (peak-weighted):")
print(f"  Mean : {recon_error_v2.mean():.6f}  Std: {recon_error_v2.std():.6f}")
print(f"  P25  : {np.percentile(recon_error_v2, 25):.6f}")
print(f"  P50  : {np.percentile(recon_error_v2, 50):.6f}")
print(f"  P75  : {np.percentile(recon_error_v2, 75):.6f}")

# ============================================================================
# STEP 6: Multiple thresholds → binary + 3-class labels
# ============================================================================
print("\n" + "="*70)
print("STEP 6: THRESHOLD COMPARISON + 3-CLASS LABELLING")
print("="*70)

# --- Threshold 1: Otsu (automatic, full-distribution valley) ---
thresh_otsu = float(threshold_otsu(recon_error_v2))

# --- Threshold 2: P80 percentile ---
# Bottom 80% = signal (~55k), top 20% = noise (~14k) ≈ mirrors original band-4 count
thresh_p80 = float(np.percentile(recon_error_v2, 80))

# --- Threshold 3 & 4: Band-anchored (most principled) ---
# Use the new errors of spectra we KNOW are signal (band-0) and KNOW are noisy (band-4)
# to set data-driven boundaries.
band4_mask_orig = error_band == 4    # ~13,988 original worst-20%
band0_mask_orig = error_band == 0    # ~13,988 original best-20% (= training set)

# "Anything above P10 of what we know is noisy → noise" (catches 90% of band-4)
thresh_band4_p10 = float(np.percentile(recon_error_v2[band4_mask_orig], 10))

# "Anything below P90 of what we know is signal → signal" (keeps 90% of band-0)
thresh_band0_p90 = float(np.percentile(recon_error_v2[band0_mask_orig], 90))

# --- Binary labels under each threshold ---
label_otsu  = np.where(recon_error_v2 < thresh_otsu,       'signal', 'noise')
label_p80   = np.where(recon_error_v2 < thresh_p80,         'signal', 'noise')
label_band4 = np.where(recon_error_v2 < thresh_band4_p10,   'signal', 'noise')

n_sig_otsu  = int((label_otsu  == 'signal').sum())
n_noi_otsu  = int((label_otsu  == 'noise').sum())
n_sig_p80   = int((label_p80   == 'signal').sum())
n_noi_p80   = int((label_p80   == 'noise').sum())
n_sig_band4 = int((label_band4 == 'signal').sum())
n_noi_band4 = int((label_band4 == 'noise').sum())

# --- 3-class: band-anchored (signal / uncertain / noise) ---
# The two band-anchor thresholds define the "certain" regions.
# If band0_p90 < band4_p10 there is a proper gap → uncertain zone between them.
# If they cross (gap < 0) both thresholds overlap → increase to P95/P5 automatically.
if thresh_band0_p90 < thresh_band4_p10:
    thresh_3low  = thresh_band0_p90
    thresh_3high = thresh_band4_p10
else:
    # Overlap: tighten to P95 of band0 / P5 of band4
    thresh_3low  = float(np.percentile(recon_error_v2[band0_mask_orig], 95))
    thresh_3high = float(np.percentile(recon_error_v2[band4_mask_orig],  5))
    print("  NOTE: band anchors overlapped → using P95/P5 fallback")

label_3class = np.where(
    recon_error_v2 <  thresh_3low,  'signal',
    np.where(recon_error_v2 < thresh_3high, 'uncertain', 'noise')
)
n_sig_3 = int((label_3class == 'signal').sum())
n_unc_3 = int((label_3class == 'uncertain').sum())
n_noi_3 = int((label_3class == 'noise').sum())

# Primary binary label for CSV and downstream use: P80 (most balanced)
label_arr     = label_p80
sig_mask_bool = label_arr == 'signal'
noi_mask_bool = label_arr == 'noise'

sig_err_mean = recon_error_v2[sig_mask_bool].mean()
noi_err_mean = recon_error_v2[noi_mask_bool].mean()

print(f"\n  {'Method':<28} {'Threshold':>12}  {'Signal':>9}  {'Noise':>9}")
print(f"  {'-'*64}")
print(f"  {'Otsu':<28} {thresh_otsu:>12.6f}  {n_sig_otsu:>9,d}  {n_noi_otsu:>9,d}")
print(f"  {'P80  (primary)':<28} {thresh_p80:>12.6f}  {n_sig_p80:>9,d}  {n_noi_p80:>9,d}")
print(f"  {'Band-4 P10 anchor':<28} {thresh_band4_p10:>12.6f}  {n_sig_band4:>9,d}  {n_noi_band4:>9,d}")
print(f"\n  3-class (band-anchored):")
print(f"    Low  threshold (band-0 P90) : {thresh_3low:.6f}")
print(f"    High threshold (band-4 P10) : {thresh_3high:.6f}")
print(f"    Signal    (<low)            : {n_sig_3:7,d}  ({n_sig_3/N*100:.1f}%)")
print(f"    Uncertain (low – high)      : {n_unc_3:7,d}  ({n_unc_3/N*100:.1f}%)")
print(f"    Noise     (>= high)         : {n_noi_3:7,d}  ({n_noi_3/N*100:.1f}%)")
print(f"\n  Separation ratio (P80 threshold): {noi_err_mean/sig_err_mean:.1f}×")

# ============================================================================
# STEP 6b: F1 Score + Silhouette Score
# ============================================================================
print("\n" + "="*70)
print("STEP 6b: COMPUTING F1 + SILHOUETTE SCORES")
print("="*70)

# ---- F1 score  (band-0 = signal, band-4 = noise as proxy ground truth) ----
# Only evaluate on the two "anchor" bands where the assignment is unambiguous.
band04_mask = (error_band == 0) | (error_band == 4)
y_true_04   = np.where(error_band[band04_mask] == 0, 'signal', 'noise')

f1_results = {}
for method_name, pred_arr in [
    ('otsu',   label_otsu),
    ('p80',    label_p80),
    ('band4',  label_band4),
    ('3class_binary',  # collapse uncertain → signal for binary F1
     np.where(label_3class[band04_mask] == 'noise', 'noise', 'signal')),
]:
    y_pred = pred_arr[band04_mask] if method_name != '3class_binary' else pred_arr
    f1     = float(f1_score(y_true_04, y_pred, pos_label='signal',
                            average='binary', zero_division=0))
    f1_results[f'f1_{method_name}'] = f1
    print(f"  F1 ({method_name:16s}): {f1:.4f}  "
          f"[n_band04={band04_mask.sum():,}]")

# Full report for primary (P80)
print("\n  Classification Report — P80 threshold (band0=signal, band4=noise):")
print(classification_report(y_true_04, label_p80[band04_mask],
                            target_names=['signal', 'noise'], zero_division=0))

# ---- Silhouette score  (subsample 8 000 or full, whichever is smaller) ----
rng_sil  = np.random.default_rng(0)
n_sil    = min(8000, N)
idx_sil  = rng_sil.choice(N, size=n_sil, replace=False)

# Use the new model's latent space (mean-pooled, standardised) as the feature
scaler_sil = StandardScaler()
X_sil      = scaler_sil.fit_transform(new_latents[idx_sil])

# Binary silhouette (P80)
lbl_sil_bin = label_p80[idx_sil]
n_cls_bin   = len(np.unique(lbl_sil_bin))
if n_cls_bin > 1:
    sil_binary   = float(silhouette_score(X_sil, lbl_sil_bin, metric='euclidean'))
    print(f"\n  Silhouette (binary P80, n={n_sil:,}): {sil_binary:.4f}")
else:
    sil_binary = None
    print("  Silhouette (binary): skipped — only one class present")

# 3-class silhouette
lbl_sil_3   = label_3class[idx_sil]
n_cls_3     = len(np.unique(lbl_sil_3))
if n_cls_3 > 1:
    sil_3class   = float(silhouette_score(X_sil, lbl_sil_3, metric='euclidean'))
    print(f"  Silhouette (3-class, n={n_sil:,}):    {sil_3class:.4f}")
else:
    sil_3class = None
    print("  Silhouette (3-class): skipped — fewer than 2 classes in subsample")

# ============================================================================
# STEP 7: Output plots
# ============================================================================
print("\n" + "="*70)
print("STEP 7: GENERATING OUTPUT PLOTS")
print("="*70)

# --- 03: Error histogram with all 3 thresholds ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].hist(recon_error_v2, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
axes[0].axvline(thresh_otsu,       color='red',    lw=2, linestyle='--',
                label=f'Otsu       : {thresh_otsu:.5f}  → {n_noi_otsu:,} noise')
axes[0].axvline(thresh_p80,        color='orange', lw=2, linestyle='-.',
                label=f'P80        : {thresh_p80:.5f}  → {n_noi_p80:,} noise')
axes[0].axvline(thresh_band4_p10,  color='purple', lw=2, linestyle=':',
                label=f'Band-4 P10 : {thresh_band4_p10:.5f}  → {n_noi_band4:,} noise')
axes[0].set_xlabel('Peak-Weighted Recon Error', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Error Distribution — All Thresholds Compared', fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Zoomed in to show the bulk of the distribution
p98 = np.percentile(recon_error_v2, 98)
axes[1].hist(recon_error_v2[recon_error_v2 <= p98], bins=100,
             color='steelblue', edgecolor='none', alpha=0.7)
axes[1].axvline(thresh_otsu,       color='red',    lw=2, linestyle='--', label='Otsu')
axes[1].axvline(thresh_p80,        color='orange', lw=2, linestyle='-.', label='P80')
axes[1].axvline(thresh_band4_p10,  color='purple', lw=2, linestyle=':',  label='Band-4 P10')
axes[1].axvspan(thresh_3low, thresh_3high, alpha=0.12, color='gold',
                label=f'Uncertain zone')
axes[1].set_xlabel('Peak-Weighted Recon Error (zoomed to P98)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Error Distribution — Zoomed (P98 cap)', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Threshold Comparison: Otsu vs P80 vs Band-4 P10', fontsize=13,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/03_error_dist_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_error_dist_v2.png")

# --- 04: VQ min-dist for signal vs noise (P80) ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(vq_min_dist[sig_mask_bool], bins=60,
        color='steelblue', alpha=0.7, label=f'Signal ({n_sig_p80:,})', edgecolor='none')
ax.hist(vq_min_dist[noi_mask_bool], bins=60,
        color='firebrick',  alpha=0.7, label=f'Noise  ({n_noi_p80:,})', edgecolor='none')
ax.set_xlabel('Mean Min VQ Distance', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('VQ Codebook Min-Distance: Signal vs Noise (P80 threshold)',
             fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_vq_dist_dist.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_vq_dist_dist.png")

# --- 05: t-SNE binary (P80) ---
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(latents_tsne[noi_mask_bool, 0], latents_tsne[noi_mask_bool, 1],
           c='firebrick', alpha=0.4, s=8, edgecolors='none',
           label=f'Noise ({n_noi_p80:,})')
ax.scatter(latents_tsne[sig_mask_bool, 0], latents_tsne[sig_mask_bool, 1],
           c='steelblue', alpha=0.4, s=8, edgecolors='none',
           label=f'Signal ({n_sig_p80:,})')
ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax.set_title('VQ-VAE Latent Space (t-SNE)\nSignal / Noise — P80 threshold',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{output_dir}/05_tsne_signal_noise.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 05_tsne_signal_noise.png")

# --- 05b: t-SNE 3-class (signal / uncertain / noise) ---
unc_mask  = label_3class == 'uncertain'
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(latents_tsne[label_3class == 'noise', 0],
           latents_tsne[label_3class == 'noise', 1],
           c='firebrick', alpha=0.4, s=8, edgecolors='none',
           label=f'Noise ({n_noi_3:,})')
ax.scatter(latents_tsne[unc_mask, 0], latents_tsne[unc_mask, 1],
           c='gold', alpha=0.5, s=8, edgecolors='none',
           label=f'Uncertain ({n_unc_3:,})')
ax.scatter(latents_tsne[label_3class == 'signal', 0],
           latents_tsne[label_3class == 'signal', 1],
           c='steelblue', alpha=0.4, s=8, edgecolors='none',
           label=f'Signal ({n_sig_3:,})')
ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax.set_title('VQ-VAE Latent Space (t-SNE)\n3-Class: Signal / Uncertain / Noise',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=3)
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{output_dir}/05b_tsne_3class.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 05b_tsne_3class.png")

# --- 06: t-SNE continuous error ---
fig, ax = plt.subplots(figsize=(12, 10))
sc = ax.scatter(
    latents_tsne[:, 0], latents_tsne[:, 1],
    c=recon_error_v2, cmap='viridis', alpha=0.5, s=8, edgecolors='none',
    vmax=np.percentile(recon_error_v2, 99),   # cap colourbar at P99
)
cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label('Peak-Weighted Recon Error (capped at P99)', fontsize=12)
ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
ax.set_title('VQ-VAE Latent Space (t-SNE)\nColoured by New Reconstruction Error',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f'{output_dir}/06_tsne_signal_noise_error.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 06_tsne_signal_noise_error.png")

# --- 07: Overlays — 4 signal + 4 uncertain + 4 noise ---
rng      = np.random.default_rng(42)
samp_sig = rng.choice(np.where(label_3class == 'signal')[0],    size=4, replace=False)
samp_unc = rng.choice(np.where(label_3class == 'uncertain')[0], size=min(4, n_unc_3),
                       replace=False)
samp_noi = rng.choice(np.where(label_3class == 'noise')[0],     size=4, replace=False)

# Pad uncertain to 4 if fewer samples exist
while len(samp_unc) < 4:
    samp_unc = np.append(samp_unc, samp_unc[-1])

all_samp = np.concatenate([samp_sig, samp_unc, samp_noi])
samp_lbs = ['Signal'] * 4 + ['Uncertain'] * 4 + ['Noise'] * 4
col_map  = {
    'Signal':    'steelblue',
    'Uncertain': 'goldenrod',
    'Noise':     'firebrick',
}

fig, axes = plt.subplots(4, 3, figsize=(18, 18))
positions = [(r, c) for r in range(4) for c in range(3)]

for pos_i, (idx, lbl) in enumerate(zip(all_samp, samp_lbs)):
    r, c = positions[pos_i]
    ax   = axes[r, c]
    c1   = col_map[lbl]
    ax.plot(X_processed[idx], lw=2.0, color=c1,    alpha=0.85, label='Original')
    ax.plot(X_recon_v2[idx],  lw=1.5, color='black', alpha=0.85, label='Reconstructed',
            linestyle='--')
    ax.set_xlabel('Mass Channel', fontsize=9)
    ax.set_ylabel('Intensity',    fontsize=9)
    ax.set_title(f'{lbl} — Spectrum {idx}\n'
                 f'New MSE={recon_error_v2[idx]:.5f}  '
                 f'Orig MSE={recon_errors[idx]:.5f}',
                 fontsize=9, fontweight='bold', color=c1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(
    'Original vs Reconstructed — 4 Signal (row 1) + 4 Uncertain (row 2–3) + 4 Noise (row 4)',
    fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/07_signal_noise_overlays.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_signal_noise_overlays.png")

# --- 08: Error scatter + 3-violin per class ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

col_map2 = {'signal': 'steelblue', 'noise': 'firebrick', 'uncertain': 'goldenrod'}
for lbl in ['signal', 'uncertain', 'noise']:
    m = label_3class == lbl
    axes[0].scatter(recon_errors[m], recon_error_v2[m],
                    c=col_map2[lbl], alpha=0.25, s=4, label=lbl, edgecolors='none')
# Draw threshold lines
axes[0].axhline(thresh_otsu,       color='red',    lw=1.5, ls='--', alpha=0.7,
                label=f'Otsu {thresh_otsu:.5f}')
axes[0].axhline(thresh_p80,        color='orange', lw=1.5, ls='-.', alpha=0.7,
                label=f'P80  {thresh_p80:.5f}')
axes[0].axhline(thresh_band4_p10,  color='purple', lw=1.5, ls=':',  alpha=0.7,
                label=f'Band-4 P10 {thresh_band4_p10:.5f}')
axes[0].set_xlabel('Original MSE (full-data model)', fontsize=11)
axes[0].set_ylabel('New Peak-Weighted MSE (signal-only model)', fontsize=11)
axes[0].set_title('Original vs New Recon Error\n(3-class colours, all 3 threshold lines)',
                  fontweight='bold')
axes[0].legend(fontsize=8, markerscale=3)
axes[0].grid(True, alpha=0.3)

parts = axes[1].violinplot(
    [recon_error_v2[label_3class == 'signal'],
     recon_error_v2[label_3class == 'uncertain'],
     recon_error_v2[label_3class == 'noise']],
    positions=[1, 2, 3], showmedians=True, showextrema=True,
)
for pc, col in zip(parts['bodies'], ['steelblue', 'goldenrod', 'firebrick']):
    pc.set_facecolor(col); pc.set_alpha(0.6)
axes[1].set_xticks([1, 2, 3])
axes[1].set_xticklabels(['Signal', 'Uncertain', 'Noise'], fontsize=12)
axes[1].set_ylabel('Peak-Weighted Recon Error', fontsize=11)
axes[1].set_title('Error Distribution by Class (3-class)', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/08_error_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 08_error_comparison.png")

# ============================================================================
# STEP 8: Save labels CSV and arrays
# ============================================================================
print("\n" + "="*70)
print("STEP 8: SAVING OUTPUTS")
print("="*70)

band_name_map = {0: 'band_0', 1: 'band_1', 2: 'band_2', 3: 'band_3', 4: 'band_4'}
label_df = pd.DataFrame({
    'index':            np.arange(N),
    'original_error':   recon_errors,
    'recon_error_v2':   recon_error_v2,
    'vq_min_dist':      vq_min_dist,
    'orig_error_band':  error_band,
    'orig_error_band_name': [band_name_map.get(b, f'band_{b}') for b in error_band],
    'label_otsu':       label_otsu,
    'label_p80':        label_p80,
    'label_band4':      label_band4,
    'label_3class':     label_3class,
    'label':            label_arr,           # alias for label_p80 (primary)
    'tsne_x':           latents_tsne[:, 0],
    'tsne_y':           latents_tsne[:, 1],
})
csv_path = f'{output_dir}/signal_noise_labels.csv'
label_df.to_csv(csv_path, index=False)
print(f"Saved: signal_noise_labels.csv  ({len(label_df):,} rows)")

np.savez_compressed(
    f'{output_dir}/signal_noise_data.npz',
    recon_error_v2 = recon_error_v2,
    vq_min_dist    = vq_min_dist,
    label_arr      = label_arr.astype(str),
    label_3class   = label_3class.astype(str),
    X_recon_v2     = X_recon_v2,
)
print("Saved: signal_noise_data.npz")

# ---- Save metrics.json ----
metrics_out = {
    # Reconstruction error statistics (new model)
    'recon_error_v2_mean':  float(recon_error_v2.mean()),
    'recon_error_v2_std':   float(recon_error_v2.std()),
    'recon_error_v2_p25':   float(np.percentile(recon_error_v2, 25)),
    'recon_error_v2_p50':   float(np.percentile(recon_error_v2, 50)),
    'recon_error_v2_p75':   float(np.percentile(recon_error_v2, 75)),
    # Error by original band
    'recon_error_v2_by_band': {
        f'band_{b}': {
            'mean': float(recon_error_v2[error_band == b].mean()),
            'std':  float(recon_error_v2[error_band == b].std()),
            'p50':  float(np.percentile(recon_error_v2[error_band == b], 50)),
            'n':    int((error_band == b).sum()),
        }
        for b in sorted(np.unique(error_band))
    },
    # Threshold values
    'threshold_otsu':       thresh_otsu,
    'threshold_p80':        thresh_p80,
    'threshold_band4_p10':  thresh_band4_p10,
    'threshold_3class_low': thresh_3low,
    'threshold_3class_high':thresh_3high,
    # Counts per method
    'counts_otsu':  {'signal': int(n_sig_otsu),  'noise': int(n_noi_otsu)},
    'counts_p80':   {'signal': int(n_sig_p80),   'noise': int(n_noi_p80)},
    'counts_band4': {'signal': int(n_sig_band4),  'noise': int(n_noi_band4)},
    'counts_3class':{'signal': int(n_sig_3), 'uncertain': int(n_unc_3),
                     'noise': int(n_noi_3)},
    # F1 scores (band-0 vs band-4 as ground truth)
    **f1_results,
    # Silhouette scores
    'silhouette_binary_p80': sil_binary,
    'silhouette_3class':     sil_3class,
    'silhouette_n_subsample':n_sil,
    # Training summary
    'training_n_signal':     int(n_signal),
    'final_recon_loss':      float(history['recon'][-1]),
    'final_perplexity':      float(history['perplexity'][-1]),
    'num_embeddings':        CONFIG['num_embeddings_new'],
    'peak_weight_scale':     CONFIG['peak_weight_scale'],
}
metrics_path = f'{output_dir}/metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics_out, f, indent=2)
print(f"Saved: metrics.json")

# ============================================================================
# Summary
# ============================================================================
summary = f"""
Signal/Noise Separation Summary
{'='*70}

Previous run      : {PREV_RUN_DIR}
Model checkpoint  : {PREV_MODEL_DIR}/vqvae_model.pth
Output directory  : {output_dir}

Training set      : Band-0 only (bottom 20% MSE) — {n_signal:,} spectra
New codebook size : {CONFIG['num_embeddings_new']} codes (was 128)
Peak weight scale : {CONFIG['peak_weight_scale']} (10x at peak tops)
Training epochs   : {CONFIG['epochs']}
LR schedule       : CosineAnnealing ({CONFIG['lr']} → 1e-5)

Final perplexity  : {history['perplexity'][-1]:.2f} / {CONFIG['num_embeddings_new']}
Final recon loss  : {history['recon'][-1]:.6f}

--- Multi-Threshold Separation Results (N = {N:,}) ---

  Threshold          Value       Signal      Noise
  ----------------------------------------------------------
  Otsu (automatic) : {thresh_otsu:.6f}   {n_sig_otsu:7,d} ({n_sig_otsu/N*100:.1f}%)  {n_noi_otsu:6,d} ({n_noi_otsu/N*100:.1f}%)
  P80  (primary)   : {thresh_p80:.6f}   {n_sig_p80:7,d} ({n_sig_p80/N*100:.1f}%)  {n_noi_p80:6,d} ({n_noi_p80/N*100:.1f}%)
  Band-4 P10       : {thresh_band4_p10:.6f}   {n_sig_band4:7,d} ({n_sig_band4/N*100:.1f}%)  {n_noi_band4:6,d} ({n_noi_band4/N*100:.1f}%)

  3-Class (band-anchored):
    thresh_3low  = {thresh_3low:.6f}  (signal  < this)
    thresh_3high = {thresh_3high:.6f}  (noise   > this)
    Signal    : {n_sig_3:7,d} ({n_sig_3/N*100:.1f}%)
    Uncertain : {n_unc_3:7,d} ({n_unc_3/N*100:.1f}%)
    Noise     : {n_noi_3:7,d} ({n_noi_3/N*100:.1f}%)

  Mean error — Signal (P80): {sig_err_mean:.6f}
  Mean error — Noise  (P80): {noi_err_mean:.6f}
  Separation ratio          : {noi_err_mean / sig_err_mean:.1f}x

--- F1 Scores (band-0=signal, band-4=noise as ground truth) ---
  F1 Otsu          : {f1_results.get('f1_otsu', float('nan')):.4f}
  F1 P80           : {f1_results.get('f1_p80', float('nan')):.4f}
  F1 Band-4 P10    : {f1_results.get('f1_band4', float('nan')):.4f}
  F1 3-class binary: {f1_results.get('f1_3class_binary', float('nan')):.4f}

--- Silhouette Scores (n_subsample={n_sil:,}, signal-only latent space) ---
  Binary (P80) : {sil_binary if sil_binary is not None else 'N/A'}
  3-Class      : {sil_3class if sil_3class is not None else 'N/A'}
  (range: -1 worst → 0 overlap → +1 best)

Output files:
  01_band0_sample_spectra.png    — 20 random band-0 training spectra
  02_training_curves.png         — loss & perplexity vs epoch
  03_error_dist_v2.png           — error histogram + all 3 thresholds
  04_vq_dist_dist.png            — VQ min-distance: signal vs noise (P80)
  05_tsne_signal_noise.png       — t-SNE coloured binary signal/noise (P80)
  05b_tsne_3class.png            — t-SNE 3-class: signal/uncertain/noise
  06_tsne_signal_noise_error.png — t-SNE coloured by new error (continuous)
  07_signal_noise_overlays.png   — original vs recon: 4 each class
  08_error_comparison.png        — original vs new error scatter + 3 violins
  signal_noise_labels.csv        — all 4 label columns + error scores
  signal_noise_data.npz          — arrays for downstream analysis
  signal_only_model.pth          — saved signal-only VQ-VAE weights
"""

print(summary)
with open(f'{output_dir}/SUMMARY.txt', 'w') as f:
    f.write(summary)

print(f"\n{'='*70}")
print(f"ALL OUTPUTS SAVED TO: {output_dir}")
print(f"{'='*70}\n")
