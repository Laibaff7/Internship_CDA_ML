#!/usr/bin/env python3
"""
build_interactive_explorer.py

Builds a self-contained interactive HTML explorer for the VQ-VAE latent space.

Features:
  • Scatter plot (t-SNE / UMAP toggle) coloured by reconstruction error or error band
  • Click any point → right panel shows original vs reconstructed spectrum
  • 10,000 random samples to keep file size manageable
  • Fully self-contained HTML (Plotly loaded from CDN — requires internet on first open,
    or replace the CDN <script> tag with a local plotly.min.js path)

Usage:
  python scripts/build_interactive_explorer.py

Output:
  outputs/interactive_tsne_explorer.html
"""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "outputs" / "unknown_vqvae" / "unknown_vqvae_analysis_latest"
OUT_PATH  = BASE_DIR / "outputs" / "interactive_tsne_explorer.html"

# ─── Config ───────────────────────────────────────────────────────────────────
N_SAMPLES   = 10_000
N_CHANNELS  = 200      # downsample 1000 → 200 for manageable file size
RANDOM_SEED = 42

# ─── Load ─────────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_DIR} …")
npz = np.load(DATA_DIR / "analysis_data.npz", allow_pickle=True)
total = npz["X_processed"].shape[0]
print(f"  Total samples : {total:,}")

# ─── Sample 10 k random indices ───────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)
idx = np.sort(rng.choice(total, size=N_SAMPLES, replace=False))

tsne_x    = npz["latents_tsne"][idx, 0].astype(np.float32)
tsne_y    = npz["latents_tsne"][idx, 1].astype(np.float32)
umap_x    = npz["latents_2d"][idx, 0].astype(np.float32)
umap_y    = npz["latents_2d"][idx, 1].astype(np.float32)
recon_err = npz["recon_errors"][idx].astype(np.float32)
error_band = npz["error_band"][idx].astype(np.int32)

BAND_NAMES = ["0-20% (Best)", "20-40%", "40-60%", "60-80%", "80-100% (Worst)"]
error_band_label = [BAND_NAMES[b] for b in error_band]

# ─── Downsample spectra ───────────────────────────────────────────────────────
ch_idx  = np.linspace(0, 999, N_CHANNELS, dtype=np.int32)
X_orig  = npz["X_processed"][idx][:, ch_idx].astype(np.float32)  # (N, 200)
X_recon = npz["X_recon"][idx][:, ch_idx].astype(np.float32)      # (N, 200)
print(f"  Sampled {N_SAMPLES:,} points; spectra downsampled to {N_CHANNELS} channels")

# ─── Base64-encode spectra (Float32) ─────────────────────────────────────────
def to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).flatten().tobytes()).decode("ascii")

orig_b64  = to_b64(X_orig)
recon_b64 = to_b64(X_recon)
print(f"  Encoded sizes : orig={len(orig_b64)/1e6:.1f} MB  recon={len(recon_b64)/1e6:.1f} MB")

# ─── Bundle everything into a compact JSON blob ───────────────────────────────
data_blob = {
    "n_samples":        N_SAMPLES,
    "n_channels":       N_CHANNELS,
    "channel_indices":  ch_idx.tolist(),
    "tsne_x":           tsne_x.tolist(),
    "tsne_y":           tsne_y.tolist(),
    "umap_x":           umap_x.tolist(),
    "umap_y":           umap_y.tolist(),
    "recon_errors":     recon_err.tolist(),
    "error_band":       error_band.tolist(),
    "error_band_label": error_band_label,
    "orig_b64":         orig_b64,
    "recon_b64":        recon_b64,
}
data_json = json.dumps(data_blob, separators=(",", ":"))
print(f"  Embedded JSON  : {len(data_json)/1e6:.1f} MB")

# ─── HTML ─────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VQ-VAE Latent Space Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{height:100%;overflow:hidden;}}
  body{{
    font-family:"Segoe UI",system-ui,Arial,sans-serif;
    background:#0d0f1a;color:#d8dce8;
    display:flex;flex-direction:column;
  }}

  /* ── Header ── */
  #hdr{{
    flex-shrink:0;
    display:flex;align-items:center;gap:14px;
    padding:9px 18px;
    background:#111422;
    border-bottom:1px solid #232640;
  }}
  #hdr h1{{font-size:1rem;font-weight:700;color:#7fa8f7;letter-spacing:.02em;}}
  .sub{{font-size:.75rem;color:#555e7a;}}
  .ctrls{{display:flex;gap:6px;margin-left:auto;align-items:center;}}
  .lbl{{font-size:.72rem;color:#555e7a;}}
  .sep{{width:1px;height:18px;background:#232640;margin:0 4px;}}
  .btn{{
    padding:4px 11px;border-radius:5px;
    border:1px solid #2d3355;background:#181b2e;
    color:#9ba8d0;font-size:.72rem;cursor:pointer;
    transition:background .15s,color .15s;
  }}
  .btn:hover{{background:#222744;}}
  .btn.on{{background:#2e3d7a;border-color:#4a5da8;color:#c0ccff;}}

  /* ── Main layout ── */
  #main{{
    flex:1;display:flex;min-height:0;
    border-top:0;
  }}
  #left{{
    flex:0 0 60%;display:flex;flex-direction:column;
    border-right:1px solid #232640;
  }}
  #right{{
    flex:1;display:flex;flex-direction:column;
    min-width:0;
  }}
  .ptitle{{
    flex-shrink:0;
    font-size:.65rem;letter-spacing:.08em;text-transform:uppercase;
    padding:5px 12px;background:#0f1120;
    color:#454d6a;border-bottom:1px solid #1c1f35;
  }}
  #scatter{{flex:1;min-height:0;}}
  #spec-wrap{{flex:1;min-height:0;display:flex;flex-direction:column;gap:0;}}
  #spec-orig{{flex:1;min-height:0;border-bottom:1px solid #1c1f35;}}
  #spec-overlay{{flex:1;min-height:0;}}
  #info{{
    flex-shrink:0;
    padding:6px 14px;background:#0f1120;
    border-top:1px solid #1c1f35;
    font-size:.73rem;color:#6b7280;min-height:28px;
  }}
  #hint{{
    display:flex;flex-direction:column;
    align-items:center;justify-content:center;
    height:100%;gap:10px;
    color:#2a3050;font-size:.88rem;
    pointer-events:none;
  }}
  #hint svg{{width:44px;height:44px;}}
</style>
</head>
<body>

<div id="hdr">
  <h1>VQ-VAE Latent Space Explorer</h1>
  <span class="sub">n={N_SAMPLES:,} random samples &nbsp;·&nbsp; {N_CHANNELS}-pt spectra</span>
  <div class="ctrls">
    <span class="lbl">Projection</span>
    <button class="btn on" id="bt-tsne"  onclick="setProj('tsne')">t-SNE</button>
    <button class="btn"    id="bt-umap"  onclick="setProj('umap')">UMAP</button>
    <div class="sep"></div>
    <span class="lbl">Colour</span>
    <button class="btn on" id="bt-cont"  onclick="setCol('cont')">Recon Error</button>
    <button class="btn"    id="bt-band"  onclick="setCol('band')">Error Band</button>
    <div class="sep"></div>
    <span class="lbl">Colormap</span>
    <button class="btn on" id="bt-cmap-def" onclick="setCmap('default')">Default</button>
    <button class="btn"    id="bt-cmap-cb"  onclick="setCmap('cb')">Colorblind</button>
  </div>
</div>

<div id="main">
  <div id="left">
    <div class="ptitle">Latent Space — click a point to inspect its spectrum</div>
    <div id="scatter"></div>
  </div>
  <div id="right">
    <div class="ptitle">Spectrum Viewer</div>
    <div id="spec-wrap">
      <div id="spec-orig">
        <div id="hint">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Click any point in the scatter plot
        </div>
      </div>
      <div id="spec-overlay"></div>
    </div>
    <div id="info">—</div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════════
//  Embedded data
// ═══════════════════════════════════════════════════════════════
const D = {data_json};

// ═══════════════════════════════════════════════════════════════
//  Decode flat Float32 blobs
// ═══════════════════════════════════════════════════════════════
function decodeF32(b64) {{
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
const origFlat  = decodeF32(D.orig_b64);
const reconFlat = decodeF32(D.recon_b64);
const N = D.n_samples, C = D.n_channels;

function spectrum(flat, i) {{
  return Array.from(flat.subarray(i * C, (i + 1) * C));
}}

// ═══════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════
let proj = 'tsne', col = 'cont', cmap = 'default';
let specInited = false;
const scDiv    = document.getElementById('scatter');
const spOrig   = document.getElementById('spec-orig');
const spOverlay= document.getElementById('spec-overlay');

// Default palette
const BAND_COLORS_DEF = ['#27ae60','#3498db','#f1c40f','#e67e22','#e74c3c'];
const CSCALE_DEF      = 'Plasma';
// Okabe-Ito colorblind-safe palette
const BAND_COLORS_CB  = ['#009E73','#56B4E9','#F0E442','#E69F00','#D55E00'];
const CSCALE_CB       = 'Cividis';

const BAND_NAMES = ['0-20% (Best)','20-40%','40-60%','60-80%','80-100% (Worst)'];

function activeBandColors() {{ return cmap === 'cb' ? BAND_COLORS_CB : BAND_COLORS_DEF; }}
function activeCscale()     {{ return cmap === 'cb' ? CSCALE_CB      : CSCALE_DEF;      }}

// ═══════════════════════════════════════════════════════════════
//  Scatter
// ═══════════════════════════════════════════════════════════════
function scatterTraces() {{
  const xs = proj === 'tsne' ? D.tsne_x : D.umap_x;
  const ys = proj === 'tsne' ? D.tsne_y : D.umap_y;
  const cd  = D.recon_errors.map((e, i) => [i, e, D.error_band_label[i]]);
  const htpl = '<b>sample %{{customdata[0]}}</b><br>MSE: %{{customdata[1]:.3e}}<br>Band: %{{customdata[2]}}<extra></extra>';

  if (col === 'cont') {{
    const logE = D.recon_errors.map(v => Math.log10(Math.max(v, 1e-12)));
    return [{{
      type:'scattergl', mode:'markers',
      x:xs, y:ys,
      marker:{{
        size:3, opacity:.72,
        color:logE,
        colorscale:activeCscale(),
        showscale:true,
        colorbar:{{
          title:{{text:'log₁₀(MSE)',side:'right',font:{{color:'#7a82a0',size:10}}}},
          tickfont:{{color:'#7a82a0',size:9}},
          bgcolor:'rgba(0,0,0,0)',
          bordercolor:'rgba(0,0,0,0)',
          len:.75, thickness:14,
        }},
      }},
      customdata:cd,
      hovertemplate:htpl,
    }}];
  }} else {{
    return BAND_NAMES.map((name, b) => {{
      const bx=[],by=[],bcd=[];
      D.error_band.forEach((eb,i)=>{{
        if(eb===b){{bx.push(xs[i]);by.push(ys[i]);bcd.push(cd[i]);}}
      }});
      return {{
        type:'scattergl', mode:'markers', name,
        x:bx, y:by,
        marker:{{size:3,color:activeBandColors()[b],opacity:.75}},
        customdata:bcd, hovertemplate:htpl,
      }};
    }});
  }}
}}

function scatterLayout() {{
  return {{
    paper_bgcolor:'#0d0f1a', plot_bgcolor:'#0d0f1a',
    margin:{{l:45,r:col==='cont'?80:10,t:30,b:45}},
    xaxis:{{title:'',tickfont:{{color:'#454d6a',size:9}},gridcolor:'#181b2e',zeroline:false,showline:false}},
    yaxis:{{title:'',tickfont:{{color:'#454d6a',size:9}},gridcolor:'#181b2e',zeroline:false,showline:false}},
    legend:{{font:{{color:'#8a93b0',size:10}},bgcolor:'rgba(0,0,0,0)',x:.01,y:.99}},
    hovermode:'closest',
    uirevision: proj + col + cmap,
  }};
}}

function renderScatter() {{
  Plotly.react(scDiv, scatterTraces(), scatterLayout(), {{
    responsive:true,
    displayModeBar:true,
    modeBarButtonsToRemove:['lasso2d','select2d','toImage'],
    displaylogo:false,
  }});
  scDiv.on('plotly_click', onPointClick);
}}

// ═══════════════════════════════════════════════════════════════
//  Spectrum viewer
// ═══════════════════════════════════════════════════════════════
function onPointClick(evt) {{
  if (!evt?.points?.length) return;
  const pt  = evt.points[0];
  const [si, mse, band] = pt.customdata;   // sample index, recon error, band label

  const orig  = spectrum(origFlat,  si);
  const recon = spectrum(reconFlat, si);
  const xs    = D.channel_indices;

  const commonLayout = {{
    paper_bgcolor:'#0d0f1a', plot_bgcolor:'#111220',
    margin:{{l:55,r:20,t:32,b:38}},
    xaxis:{{
      title:'Channel index',
      titlefont:{{color:'#555e7a',size:10}},
      tickfont:{{color:'#454d6a',size:9}},
      gridcolor:'#181b2e',zeroline:false,
    }},
    yaxis:{{
      title:'Intensity (norm.)',
      titlefont:{{color:'#555e7a',size:10}},
      tickfont:{{color:'#454d6a',size:9}},
      gridcolor:'#181b2e',zeroline:false,
    }},
    legend:{{
      font:{{color:'#8a93b0',size:10}},
      bgcolor:'rgba(16,18,32,.8)',
      bordercolor:'#2a2d45',borderwidth:1,
      x:.01,y:.99,
    }},
    hovermode:'x unified',
  }};

  // ── Top plot: original spectrum only ──
  const layoutOrig = Object.assign({{}}, commonLayout, {{
    title:{{
      text:`Sample #${{si}} &nbsp;<span style="font-size:10px;color:#555e7a">— ${{band}} — Original</span>`,
      font:{{color:'#8a93b0',size:11}}, x:.04,
    }},
  }});

  // ── Bottom plot: original + reconstructed overlay ──
  const layoutOverlay = Object.assign({{}}, commonLayout, {{
    title:{{
      text:'Reconstructed Overlay',
      font:{{color:'#8a93b0',size:11}}, x:.04,
    }},
  }});

  if (!specInited) {{
    spOrig.innerHTML = '';
    specInited = true;
  }}

  Plotly.react(spOrig, [
    {{type:'scatter',mode:'lines',name:'Original',x:xs,y:orig,
      line:{{color:'#6eadf5',width:1.6}}}},
  ], layoutOrig, {{responsive:true,displayModeBar:false}});

  Plotly.react(spOverlay, [
    {{type:'scatter',mode:'lines',name:'Original',x:xs,y:orig,
      line:{{color:'#6eadf5',width:1.6}}}},
    {{type:'scatter',mode:'lines',name:'Reconstructed',x:xs,y:recon,
      line:{{color:'#fb923c',width:1.5,dash:'dot'}}}},
  ], layoutOverlay, {{responsive:true,displayModeBar:false}});

  document.getElementById('info').innerHTML =
    `<b style="color:#d8dce8">Sample #${{si}}</b>` +
    `&nbsp;&nbsp;MSE: <b style="color:#fb923c">${{mse.toExponential(3)}}</b>` +
    `&nbsp;&nbsp;Band: <b style="color:#7fa8f7">${{band}}</b>`;
}}

// ═══════════════════════════════════════════════════════════════
//  Toggle helpers
// ═══════════════════════════════════════════════════════════════
function setProj(p) {{
  proj = p;
  document.getElementById('bt-tsne').classList.toggle('on', p==='tsne');
  document.getElementById('bt-umap').classList.toggle('on', p==='umap');
  renderScatter();
}}
function setCol(c) {{
  col = c;
  document.getElementById('bt-cont').classList.toggle('on', c==='cont');
  document.getElementById('bt-band').classList.toggle('on', c==='band');
  renderScatter();
}}
function setCmap(m) {{
  cmap = m;
  document.getElementById('bt-cmap-def').classList.toggle('on', m==='default');
  document.getElementById('bt-cmap-cb').classList.toggle('on',  m==='cb');
  renderScatter();
}}

// ═══════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════
renderScatter();
</script>
</body>
</html>"""

# ─── Write ────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Writing {OUT_PATH} …")
with open(OUT_PATH, "w", encoding="utf-8") as fh:
    fh.write(html)

size_mb = OUT_PATH.stat().st_size / 1e6
print(f"Done!  File size: {size_mb:.1f} MB")
print(f"       {OUT_PATH}")
