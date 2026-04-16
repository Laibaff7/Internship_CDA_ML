#!/usr/bin/env python3
"""
build_unified_explorer.py
Builds a self-contained interactive HTML explorer unifying Base VQ-VAE and Signal/Noise data.
"""

import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd

def to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).flatten().tobytes()).decode("ascii")

def build_unified():
    # Load original data
    d = Path("/home/lain01/Laiba/outputs/unknown_vqvae/unknown_vqvae_analysis_latest")
    npz1 = np.load(d / "analysis_data.npz", allow_pickle=True)
    
    # Load signal noise data
    snd = Path("/home/lain01/Laiba/outputs/signal_noise/signal_noise_20260311_012752")
    df = pd.read_csv(snd / "signal_noise_labels.csv")
    npz2 = np.load(snd / "signal_noise_data.npz", allow_pickle=True)

    total = len(df)
    N_SAMPLES = min(10000, total)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(total, size=N_SAMPLES, replace=False))

    tsne_x    = npz1["latents_2d"][idx, 0].astype(np.float32)
    tsne_y    = npz1["latents_2d"][idx, 1].astype(np.float32)
    umap_x    = npz1["latents_tsne"][idx, 0].astype(np.float32)
    umap_y    = npz1["latents_tsne"][idx, 1].astype(np.float32)
    
    base_err  = npz1["recon_errors"][idx].astype(np.float32)
    base_band = npz1["error_band"][idx].astype(np.int32)
    
    BAND_NAMES = ["0-20% (Best)", "20-40%", "40-60%", "60-80%", "80-100% (Worst)"]
    base_band_label = [BAND_NAMES[b] for b in base_band]
    
    sn_err    = npz2["recon_error_v2"][idx].astype(np.float32)
    sn_3class = [str(npz2["label_3class"][i]) for i in idx]
    sn_p80    = [str(df["label_p80"].values[i]) for i in idx]

    # Crop length to 630
    L = 630
    X_orig   = npz1["X_processed"][idx, :L].astype(np.float32)
    X_recon1 = npz1["X_recon"][idx, :L].astype(np.float32)
    
    X_recon2_raw = npz2["X_recon_v2"]
    if len(X_recon2_raw.shape) == 3:
        X_recon2 = X_recon2_raw[idx, :, :L].squeeze().astype(np.float32)
    else:
        X_recon2 = X_recon2_raw[idx, :L].astype(np.float32)

    print("Encoding arrays to base64...")
    orig_b64   = to_b64(X_orig)
    recon1_b64 = to_b64(X_recon1)
    recon2_b64 = to_b64(X_recon2)

    data_blob = {
        "n_samples":       N_SAMPLES,
        "n_channels":      L,
        "channel_indices": list(range(L)),
        "tsne_x":          tsne_x.tolist(),
        "tsne_y":          tsne_y.tolist(),
        "umap_x":          umap_x.tolist(),
        "umap_y":          umap_y.tolist(),
        "base_err":        base_err.tolist(),
        "base_band":       base_band.tolist(),
        "base_band_label": base_band_label,
        "sn_err":          sn_err.tolist(),
        "sn_3class":       sn_3class,
        "sn_p80":          sn_p80,
        "orig_b64":        orig_b64,
        "recon1_b64":      recon1_b64,
        "recon2_b64":      recon2_b64,
    }
    
    data_json = json.dumps(data_blob, separators=(",", ":"))
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Unified Latent Space Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  html,body{{height:100%;overflow:hidden;}}
  body{{
    font-family:"Segoe UI",system-ui,Arial,sans-serif;
    background:#0d0f1a;color:#d8dce8;
    display:flex;flex-direction:column;
  }}
  #hdr{{
    flex-shrink:0;
    display:flex;align-items:center;gap:14px;
    padding:9px 18px;
    background:#111422;
    border-bottom:1px solid #232640;
  }}
  #hdr h1{{font-size:1rem;font-weight:700;color:#7fa8f7;letter-spacing:.02em;}}
  .sub{{font-size:.75rem;color:#555e7a;}}
  .ctrls{{display:flex;gap:6px;margin-left:auto;align-items:center;flex-wrap:wrap;}}
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
  #main{{flex:1;display:flex;min-height:0;}}
  #left{{
    flex:0 0 60%;display:flex;flex-direction:column;
    border-right:1px solid #232640;
  }}
  #right{{flex:1;display:flex;flex-direction:column;min-width:0;}}
  .ptitle{{
    flex-shrink:0;font-size:.65rem;letter-spacing:.08em;text-transform:uppercase;
    padding:5px 12px;background:#0f1120;color:#454d6a;border-bottom:1px solid #1c1f35;
  }}
  #scatter{{flex:1;min-height:0;}}
  #spec-wrap{{flex:1;min-height:0;display:flex;flex-direction:column;gap:0;}}
  #spec-orig{{flex:1;min-height:0;border-bottom:1px solid #1c1f35;}}
  #spec-overlay{{flex:1;min-height:0;}}
  #info{{
    flex-shrink:0;padding:6px 14px;background:#0f1120;
    border-top:1px solid #1c1f35;font-size:.73rem;color:#6b7280;min-height:28px;
  }}
  #hint{{
    display:flex;flex-direction:column;align-items:center;justify-content:center;
    height:100%;gap:10px;color:#2a3050;font-size:.88rem;pointer-events:none;
  }}
  #hint svg{{width:44px;height:44px;}}
</style>
</head>
<body>

<div id="hdr">
  <h1>Unified Latent Space Explorer</h1>
  <span class="sub">n={N_SAMPLES:,} random samples &nbsp;·&nbsp; {L}-pt spectra</span>
  <div class="ctrls">
    <span class="lbl">Proj.</span>
    <button class="btn on" id="bt-tsne" onclick="setProj('tsne')">t-SNE</button>
    <button class="btn"    id="bt-umap" onclick="setProj('umap')">UMAP</button>
    <div class="sep"></div>
    <span class="lbl">Colour</span>
    <button class="btn on" id="bt-base-err" onclick="setCol('base-err')">Base Err</button>
    <button class="btn"    id="bt-base-band" onclick="setCol('base-band')">Base Band</button>
    <button class="btn"    id="bt-3class" onclick="setCol('3class')">3-Class</button>
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
const D = {data_json};

function decodeF32(b64) {{
  const bin = window.atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8  = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}}
const origFlat   = decodeF32(D.orig_b64);
const recon1Flat = decodeF32(D.recon1_b64);
const recon2Flat = decodeF32(D.recon2_b64);
const N = D.n_samples, C = D.n_channels;

function spectrum(flat, i) {{
  return Array.from(flat.subarray(i * C, (i + 1) * C));
}}

let proj = 'tsne', col = 'base-err', cmap = 'default';
let specInited = false;
const scDiv     = document.getElementById('scatter');
const spOrig    = document.getElementById('spec-orig');
const spOverlay = document.getElementById('spec-overlay');

// Palettes
const BAND_NAMES      = ['0-20% (Best)','20-40%','40-60%','60-80%','80-100% (Worst)'];
const BAND_COLORS_DEF = ['#27ae60','#3498db','#f1c40f','#e67e22','#e74c3c'];
const BAND_COLORS_CB  = ['#009E73','#56B4E9','#F0E442','#E69F00','#D55E00'];

const CLASS3_KEYS = ['signal','uncertain','noise'];
const CLASS3_DISP = ['Signal','Uncertain','Noise'];
const CLASS3_DEF  = ['#5b9bd5','#f0c040','#c0392b'];  
const CLASS3_CB   = ['#0077BB','#EE7733','#EE3377'];  

const P80_KEYS = ['signal','noise', 'True', 'False']; 
const P80_DISP = ['Signal','Noise', 'Signal', 'Noise'];
const P80_DEF  = ['#5b9bd5','#c0392b', '#5b9bd5','#c0392b'];
const P80_CB   = ['#0077BB','#EE3377', '#0077BB','#EE3377'];

const CSCALE_DEF = 'Plasma';
const CSCALE_CB  = 'Cividis';

function activeBand()    {{ return cmap === 'cb' ? BAND_COLORS_CB : BAND_COLORS_DEF; }}
function activeClass3()  {{ return cmap === 'cb' ? CLASS3_CB      : CLASS3_DEF;  }}
function activeP80()     {{ return cmap === 'cb' ? P80_CB         : P80_DEF;     }}
function activeCscale()  {{ return cmap === 'cb' ? CSCALE_CB      : CSCALE_DEF;  }}

function scatterTraces() {{
  const xs = proj === 'tsne' ? D.tsne_x : D.umap_x;
  const ys = proj === 'tsne' ? D.tsne_y : D.umap_y;
  
  // Customdata holds info for hover & click
  // [si, base_err, base_band, sn_err, sn_3class, sn_p80]
  const cd = D.base_err.map((e,i) => [i, e, D.base_band_label[i], D.sn_err[i], D.sn_3class[i], D.sn_p80[i]]);
  
  let htpl = '<b>sample %{{customdata[0]}}</b><br>';
  
  if (col === 'base-err' || col === 'sn-err') {{
    const errs = col === 'base-err' ? D.base_err : D.sn_err;
    const logE = errs.map(v => Math.log10(Math.max(v, 1e-12)));
    htpl += 'MSE: %{{customdata[' + (col === 'base-err' ? 1 : 3) + ']:.3e}}<br>';
    htpl += 'Class: %{{customdata[4]}}<extra></extra>';
    
    return [{{
      type:'scattergl', mode:'markers', x:xs, y:ys,
      marker:{{
        size:3, opacity:.72, color:logE, colorscale:activeCscale(), showscale:true,
        colorbar:{{
          title:{{text:'log₁₀(MSE)',side:'right',font:{{color:'#7a82a0',size:10}}}},
          tickfont:{{color:'#7a82a0',size:9}},
          bgcolor:'rgba(0,0,0,0)', bordercolor:'rgba(0,0,0,0)',
          len:.75, thickness:14,
        }},
      }},
      customdata:cd, hovertemplate:htpl,
    }}];
  }}
  
  if (col === 'base-band') {{
    htpl += 'Band: %{{customdata[2]}}<br>MSE: %{{customdata[1]:.3e}}<extra></extra>';
    return BAND_NAMES.map((name, b) => {{
      const bx=[], by=[], bcd=[];
      D.base_band.forEach((eb,i) => {{
        if(eb===b){{ bx.push(xs[i]); by.push(ys[i]); bcd.push(cd[i]); }}
      }});
      return {{
        type:'scattergl', mode:'markers', name, x:bx, y:by,
        marker:{{size:3, color:activeBand()[b], opacity:.75}},
        customdata:bcd, hovertemplate:htpl,
      }};
    }});
  }}
  
  if (col === '3class') {{
    htpl += 'Class: %{{customdata[4]}}<br>Signal MSE: %{{customdata[3]:.3e}}<extra></extra>';
    return CLASS3_KEYS.map((key, ci) => {{
      const bx=[], by=[], bcd=[];
      D.sn_3class.forEach((lbl,i) => {{
        if(lbl.toLowerCase() === key.toLowerCase()) {{ bx.push(xs[i]); by.push(ys[i]); bcd.push(cd[i]); }}
      }});
      return {{
        type:'scattergl', mode:'markers', name:CLASS3_DISP[ci], x:bx, y:by,
        marker:{{size:3, color:activeClass3()[ci], opacity:.75}},
        customdata:bcd, hovertemplate:htpl,
      }};
    }});
  }}
  
  if (col === 'p80') {{
    htpl += 'P80: %{{customdata[5]}}<br>Signal MSE: %{{customdata[3]:.3e}}<extra></extra>';
    // Combine True/False or Signal/Noise mapping
    return ['Signal', 'Noise'].map((disp_name, ci) => {{
      const bx=[], by=[], bcd=[];
      D.sn_p80.forEach((lbl,i) => {{
        const lower = lbl.toLowerCase();
        const isOpt = (ci===0 && (lower==='signal'||lower==='true')) || (ci===1 && (lower==='noise'||lower==='false'));
        if(isOpt) {{ bx.push(xs[i]); by.push(ys[i]); bcd.push(cd[i]); }}
      }});
      return {{
        type:'scattergl', mode:'markers', name:disp_name, x:bx, y:by,
        marker:{{size:3, color:activeP80()[ci], opacity:.75}},
        customdata:bcd, hovertemplate:htpl,
      }};
    }});
  }}
}}

function scatterLayout() {{
  const isCont = (col === 'base-err' || col === 'sn-err');
  return {{
    paper_bgcolor:'#0d0f1a', plot_bgcolor:'#0d0f1a',
    margin:{{l:45, r:isCont?80:10, t:30, b:45}},
    xaxis:{{title:proj.toUpperCase() + ' 1', titlefont:{{color:'#555e7a',size:10}}, tickfont:{{color:'#454d6a',size:9}}, gridcolor:'#181b2e', zeroline:false, showline:false}},
    yaxis:{{title:proj.toUpperCase() + ' 2', titlefont:{{color:'#555e7a',size:10}}, tickfont:{{color:'#454d6a',size:9}}, gridcolor:'#181b2e', zeroline:false, showline:false}},
    legend:{{font:{{color:'#8a93b0',size:10}}, bgcolor:'rgba(0,0,0,0)', x:.01, y:.99}},
    hovermode:'closest',
    uirevision: proj + col + cmap,
  }};
}}

function renderScatter() {{
  Plotly.react(scDiv, scatterTraces(), scatterLayout(), {{
    responsive:true, displayModeBar:true, modeBarButtonsToRemove:['lasso2d','select2d','toImage'], displaylogo:false,
  }});
  scDiv.on('plotly_click', onPointClick);
}}

function onPointClick(evt) {{
  if (!evt?.points?.length) return;
  const pt = evt.points[0];
  const [si, base_err, base_band_l, sn_err, sn_3class, sn_p80] = pt.customdata;

  const orig   = spectrum(origFlat,   si);
  const recon1 = spectrum(recon1Flat, si);
  const recon2 = spectrum(recon2Flat, si);
  const xs     = D.channel_indices;

  const commonLayout = {{
    paper_bgcolor:'#0d0f1a', plot_bgcolor:'#111220',
    margin:{{l:55, r:20, t:32, b:38}},
    xaxis:{{ title:'m/z', titlefont:{{color:'#555e7a',size:10}}, tickfont:{{color:'#454d6a',size:9}}, gridcolor:'#181b2e', zeroline:false }},
    yaxis:{{ title:'Intensity', titlefont:{{color:'#555e7a',size:10}}, tickfont:{{color:'#454d6a',size:9}}, gridcolor:'#181b2e', zeroline:false }},
    legend:{{ font:{{color:'#8a93b0',size:10}}, bgcolor:'rgba(16,18,32,.8)', bordercolor:'#2a2d45', borderwidth:1, x:.01, y:.99 }},
    hovermode:'x unified',
  }};

  const layoutOrig = Object.assign({{}}, commonLayout, {{
    title:{{
      text:`Sample #${{si}} &nbsp;<span style="font-size:10px;color:#555e7a">— ${{sn_3class}} | ${{base_band_l}}</span>`,
      font:{{color:'#8a93b0',size:11}}, x:.04,
    }},
  }});

  const layoutOverlay = Object.assign({{}}, commonLayout, {{
    title:{{ text:'Reconstructions Overlay', font:{{color:'#8a93b0',size:11}}, x:.04 }},
  }});

  if (!specInited) {{ spOrig.innerHTML = ''; specInited = true; }}

  Plotly.react(spOrig, [
    {{type:'scatter', mode:'lines', name:'Original', x:xs, y:orig, line:{{color:'#6eadf5', width:1.6}}}},
  ], layoutOrig, {{responsive:true, displayModeBar:false}});

  Plotly.react(spOverlay, [
    {{type:'scatter', mode:'lines', name:'Original', x:xs, y:orig, line:{{color:'#6eadf5', width:1.6, opacity:0.5}}}},
    {{type:'scatter', mode:'lines', name:'Base Recon', x:xs, y:recon1, line:{{color:'#cccccc', width:1.5, dash:'dash'}}}},
    {{type:'scatter', mode:'lines', name:'Signal Recon', x:xs, y:recon2, line:{{color:'#fb923c', width:1.5, dash:'dot'}}}},
  ], layoutOverlay, {{responsive:true, displayModeBar:false}});

  let clsColor = '#e05555';
  if(sn_3class.toLowerCase()==='signal') clsColor = '#6eadf5';
  if(sn_3class.toLowerCase()==='uncertain') clsColor = '#f0c040';

  document.getElementById('info').innerHTML =
    `<b style="color:#d8dce8">Sample #${{si}}</b>` +
    `&nbsp;&nbsp;Class: <b style="color:${{clsColor}}">${{sn_3class}}</b>` +
    `&nbsp;&nbsp;Base MSE: <b style="color:#cccccc">${{base_err.toExponential(3)}}</b>` +
    `&nbsp;&nbsp;Signal MSE: <b style="color:#fb923c">${{sn_err.toExponential(3)}}</b>`;
}}

function setProj(p) {{
  proj = p;
  document.getElementById('bt-tsne').classList.toggle('on', p==='tsne');
  document.getElementById('bt-umap').classList.toggle('on', p==='umap');
  renderScatter();
}}
function setCol(c) {{
  col = c;
  ['base-err','base-band','3class'].forEach(x => {{
    document.getElementById('bt-'+x).classList.toggle('on', c===x);
  }});
  renderScatter();
}}
function setCmap(m) {{
  cmap = m;
  document.getElementById('bt-cmap-def').classList.toggle('on', m==='default');
  document.getElementById('bt-cmap-cb' ).classList.toggle('on', m==='cb');
  renderScatter();
}}

renderScatter();
</script>
</body>
</html>"""

    out = "/home/lain01/Laiba/outputs/unified_explorer.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Done generating {out}")

if __name__ == "__main__":
    build_unified()
