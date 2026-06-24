"""Generate interactive HTML peak review reports using Plotly.js."""

import json
from pathlib import Path
import numpy as np


def generate_peak_report(
    found_peaks,
    activities,
    dose_rates,
    energies,
    counts,
    output_path,
    title="Peak Review",
):
    """Generate an interactive HTML page for reviewing peak fits.

    Parameters
    ----------
    found_peaks : dict
        Fitted peaks dictionary (as returned by fit_peak loop).
    activities : dict
        Per-peak activity info (from calculate_activity_and_dose).
    dose_rates : dict
        Per-peak dose rates.
    energies : ndarray
        Energy array for the spectrum.
    counts : ndarray
        Net (background-subtracted) counts.
    output_path : str or Path
        Path to write the HTML file.
    title : str
        Page title.
    """
    peaks_data = []
    for name in sorted(found_peaks.keys(), key=lambda x: activities.get(x, {}).get("energy", 0)):
        pk = found_peaks[name]
        act = activities.get(name, {})
        popt = pk["peak_fit"]["popt"]
        mu, sigma, height = popt
        int_lims = pk["peak_fit"]["int_lims"]
        bkg_popt = pk.get("background", {}).get("popt", [0, 0])

        # Spectrum data around the peak
        pad = max(5 * abs(sigma), 20)
        mask = (energies >= int_lims[0] - pad) & (energies <= int_lims[1] + pad)
        x_spec = energies[mask].tolist()
        y_spec = counts[mask].tolist()

        # Fitted Gaussian
        x_fit = np.linspace(int_lims[0] - pad, int_lims[1] + pad, 200)
        y_fit = (height * np.exp(-0.5 * ((x_fit - mu) / sigma) ** 2)).tolist()

        # Background line
        y_bkg = (bkg_popt[0] * x_fit + bkg_popt[1]).tolist()
        x_fit = x_fit.tolist()

        # Sideband regions for background estimation
        side_left = (mu - 6 * abs(sigma), mu - 3 * abs(sigma))
        side_right = (mu + 3 * abs(sigma), mu + 6 * abs(sigma))

        ratio = act.get("net_area", 0) / act.get("counting_net", 1) if act.get("counting_net", 0) > 0 else 0
        quality = act.get("quality", {})

        peaks_data.append({
            "name": name,
            "energy": round(mu, 2),
            "sigma": round(abs(sigma), 3),
            "height": round(height, 1),
            "net_area": round(act.get("net_area", 0), 1),
            "counting_net": round(act.get("counting_net", 0), 1),
            "ratio": round(ratio, 2),
            "significance": round(act.get("significance", 0), 1),
            "probability": round(act.get("probability", 0) * 100, 2),
            "activity": act.get("activity", 0),
            "activity_err": act.get("activity_err", 0),
            "dose": dose_rates.get(name, 0),
            "bkg_area": round(act.get("bkg_area", 0), 1),
            "chi2_red": round(act.get("chi2_red", 0), 2),
            "sigma_at_bound": act.get("sigma_at_bound", False),
            "quality": quality,
            "int_lims": [round(int_lims[0], 2), round(int_lims[1], 2)],
            "side_left": [round(side_left[0], 2), round(side_left[1], 2)],
            "side_right": [round(side_right[0], 2), round(side_right[1], 2)],
            "x_spec": x_spec,
            "y_spec": y_spec,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "y_bkg": y_bkg,
        })

    # Full spectrum data for overview
    overview = {
        "x": energies.tolist(),
        "y": counts.tolist(),
    }

    html = _build_html(peaks_data, overview, title)
    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


def _build_html(peaks_data, overview, title):
    peaks_json = json.dumps(peaks_data)
    overview_json = json.dumps(overview)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; }}
  .header {{ background: #16213e; padding: 16px 24px; border-bottom: 1px solid #0f3460; }}
  .header h1 {{ font-size: 1.4em; color: #e94560; }}
  .header .subtitle {{ font-size: 0.85em; color: #888; margin-top: 4px; }}
  .tabs {{ display: flex; gap: 0; background: #16213e; border-bottom: 1px solid #0f3460; padding: 0 24px; }}
  .tab {{ padding: 10px 20px; cursor: pointer; color: #888; border-bottom: 2px solid transparent;
          font-size: 0.9em; transition: all 0.2s; }}
  .tab:hover {{ color: #e0e0e0; }}
  .tab.active {{ color: #e94560; border-bottom-color: #e94560; }}
  .container {{ display: flex; height: calc(100vh - 110px); }}
  .sidebar {{ width: 320px; min-width: 320px; background: #16213e; border-right: 1px solid #0f3460;
              overflow-y: auto; }}
  .main {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
  .plot-area {{ flex: 1; padding: 12px; overflow: hidden; }}
  #peakPlot, #overviewPlot {{ width: 100%; height: 100%; }}
  .info-bar {{ background: #16213e; border-top: 1px solid #0f3460; padding: 12px 20px;
               display: flex; gap: 24px; flex-wrap: wrap; font-size: 0.9em; }}
  .info-bar .item {{ display: flex; gap: 6px; }}
  .info-bar .label {{ color: #888; }}
  .info-bar .value {{ color: #e94560; font-weight: 600; }}
  .info-bar .ok {{ color: #4ecca3; }}
  .info-bar .fail {{ color: #e94560; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
  th {{ position: sticky; top: 0; background: #0f3460; color: #e0e0e0; padding: 8px 10px;
        text-align: left; cursor: pointer; user-select: none; white-space: nowrap; }}
  th:hover {{ background: #1a4a8a; }}
  th .arrow {{ font-size: 0.7em; margin-left: 3px; }}
  td {{ padding: 7px 10px; border-bottom: 1px solid #1a1a3e; white-space: nowrap; }}
  tr {{ cursor: pointer; }}
  tr:hover {{ background: #1a3a5e; }}
  tr.active {{ background: #0f3460; border-left: 3px solid #e94560; }}
  .nav {{ display: flex; gap: 8px; padding: 8px 12px; background: #16213e; border-bottom: 1px solid #0f3460; }}
  .nav button {{ background: #0f3460; color: #e0e0e0; border: 1px solid #1a4a8a; padding: 6px 14px;
                border-radius: 4px; cursor: pointer; font-size: 0.85em; }}
  .nav button:hover {{ background: #1a4a8a; }}
  .nav .counter {{ padding: 6px 12px; color: #888; font-size: 0.85em; }}
  .hidden {{ display: none !important; }}
  tr.failed {{ background: #2a1a1a; opacity: 0.5; }}
  tr.failed:hover {{ background: #3a2a2a; }}
  .toggle-bar {{ display: flex; gap: 12px; align-items: center; padding: 6px 24px;
                 background: #16213e; border-bottom: 1px solid #0f3460; font-size: 0.85em; }}
  .toggle-bar label {{ color: #888; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
  .toggle-bar input[type=checkbox] {{ accent-color: #e94560; }}
</style>
</head>
<body>
<div class="header">
  <h1>{title}</h1>
  <div class="subtitle" id="subtitle"></div>
</div>
<div class="tabs">
  <div class="tab active" data-view="overview" onclick="switchView('overview')">Overview</div>
  <div class="tab" data-view="peaks" onclick="switchView('peaks')">Individual Peaks</div>
</div>

<div id="overviewView">
  <div class="toggle-bar">
    <label><input type="checkbox" id="showFailed" checked onchange="toggleFailed()"> Show failed peaks</label>
  </div>
  <div style="height: calc(100vh - 180px); padding: 12px;">
    <div id="overviewPlot" style="width:100%; height:100%;"></div>
  </div>
</div>

<div id="peaksView" class="hidden">
  <div class="container">
    <div class="sidebar">
      <table id="summaryTable">
        <thead>
          <tr>
            <th data-col="name">Isotope <span class="arrow"></span></th>
            <th data-col="energy">E (keV) <span class="arrow"></span></th>
            <th data-col="significance">Signif <span class="arrow"></span></th>
            <th data-col="ratio">F/C <span class="arrow"></span></th>
            <th data-col="chi2">χ² <span class="arrow"></span></th>
            <th data-col="quality">Quality <span class="arrow"></span></th>
            <th data-col="activity">Bq <span class="arrow"></span></th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
    <div class="main">
      <div class="nav">
        <button onclick="navigate(-1)">&larr; Prev</button>
        <span class="counter" id="navCounter"></span>
        <button onclick="navigate(1)">Next &rarr;</button>
      </div>
      <div class="plot-area">
        <div id="peakPlot"></div>
      </div>
      <div class="info-bar" id="infoBar"></div>
    </div>
  </div>
</div>

<script>
const PEAKS = {peaks_json};
const OVERVIEW = {overview_json};
let currentIdx = 0;
let sortCol = 'energy';
let sortAsc = true;
let overviewDrawn = false;

function switchView(view) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  document.getElementById('overviewView').classList.toggle('hidden', view !== 'overview');
  document.getElementById('peaksView').classList.toggle('hidden', view !== 'peaks');
  if (view === 'overview' && !overviewDrawn) {{
    drawOverview();
    overviewDrawn = true;
  }}
  if (view === 'peaks') {{
    setTimeout(() => Plotly.Plots.resize('peakPlot'), 50);
  }}
}}

function drawOverview() {{
  drawOverviewFiltered(true);
}}

function drawOverviewFiltered(showFailed) {{
  // Filter to positive counts for log scale
  const posIdx = [];
  for (let i = 0; i < OVERVIEW.y.length; i++) {{
    if (OVERVIEW.y[i] > 0) posIdx.push(i);
  }}
  const xPos = posIdx.map(i => OVERVIEW.x[i]);
  const yPos = posIdx.map(i => OVERVIEW.y[i]);

  const yMax = Math.max(...yPos);
  const yMin = Math.min(...yPos);

  const dataTrace = {{
    x: xPos, y: yPos,
    mode: 'markers', name: 'Net counts',
    marker: {{ color: '#4fc3f7', size: 2 }},
  }};

  // Fit traces: passed in green, failed in dim red
  const fitTraces = PEAKS.filter(p => showFailed || (p.quality && p.quality.pass)).map((p, i) => {{
    const passed = p.quality && p.quality.pass;
    return {{
      x: p.x_fit, y: p.y_fit.map(v => v + (p.y_bkg[Math.floor(p.y_bkg.length / 2)] || 0)),
      mode: 'lines', name: p.name + (passed ? '' : ' [FAIL]'),
      line: {{ color: passed ? '#4ecca3' : '#666', width: passed ? 1.5 : 1, dash: passed ? 'solid' : 'dot' }},
      visible: 'legendonly',
    }};
  }});

  // Annotations: only for passed peaks (or all if showFailed)
  const visiblePeaks = PEAKS.filter(p => showFailed || (p.quality && p.quality.pass));
  const annotations = visiblePeaks.map(p => ({{
    x: p.energy, y: p.height + (p.y_bkg[Math.floor(p.y_bkg.length / 2)] || 0),
    text: p.name.replace('_', '-') + ((p.quality && !p.quality.pass) ? ' [FAIL]' : ''),
    showarrow: true, arrowhead: 2, arrowsize: 0.8, arrowwidth: 1,
    ax: 0, ay: -30 - Math.random() * 20,
    font: {{ size: 9, color: (p.quality && p.quality.pass) ? '#4ecca3' : '#e94560' }},
    bgcolor: 'rgba(26,26,46,0.8)', borderpad: 2,
  }}));

  // Vertical lines: green for passed, dim for failed
  const shapes = visiblePeaks.map(p => ({{
    type: 'line', x0: p.energy, x1: p.energy, y0: 0, y1: 1,
    yref: 'paper', line: {{ color: (p.quality && p.quality.pass) ? '#4ecca3' : '#555',
                            width: (p.quality && p.quality.pass) ? 0.5 : 0.3, dash: 'dot' }},
  }}));

  const layout = {{
    margin: {{ t: 40, r: 30, b: 50, l: 60 }},
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    font: {{ color: '#e0e0e0' }},
    xaxis: {{ title: 'Energy (keV)', gridcolor: '#2a2a4e', zeroline: false }},
    yaxis: {{ title: 'Counts', gridcolor: '#2a2a4e', zeroline: false, type: 'log',
              range: [Math.log10(Math.max(1, yMin * 0.5)), Math.log10(yMax * 3)] }},
    title: {{ text: 'Spectrum Overview — click legend to show/hide fits', font: {{ size: 14, color: '#e94560' }} }},
    shapes: shapes,
    annotations: annotations,
    legend: {{ x: 1.02, y: 1, font: {{ size: 9 }} }},
  }};

  Plotly.newPlot('overviewPlot', [dataTrace, ...fitTraces], layout, {{ responsive: true }});
}}

function init() {{
  const nPass = PEAKS.filter(p => p.quality && p.quality.pass).length;
  const nFail = PEAKS.length - nPass;
  document.getElementById('subtitle').textContent =
    `${{nPass}} passed, ${{nFail}} failed of ${{PEAKS.length}} peaks | Click row or use arrows to navigate`;
  renderTable();
  selectPeak(0);
  drawOverview();
  overviewDrawn = true;
}}

function toggleFailed() {{
  const show = document.getElementById('showFailed').checked;
  drawOverviewFiltered(show);
}}

function renderTable() {{
  const sorted = [...PEAKS].sort((a, b) => {{
    let va = a[sortCol], vb = b[sortCol];
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? va - vb : vb - va;
  }});
  sorted.forEach((p, i) => p._idx = PEAKS.indexOf(p));

  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = sorted.map(p => {{
    const act = p.activity > 0 ? p.activity.toExponential(2) : '—';
    const ratStr = p.ratio > 0 ? p.ratio.toFixed(2) : '—';
    const q = p.quality || {{}};
    const qIcons = [
      q.chi2 ? 'χ' : '<span style="color:#e94560">χ</span>',
      q.sigma ? 'σ' : '<span style="color:#e94560">σ</span>',
      q.ratio ? 'R' : '<span style="color:#e94560">R</span>',
      q.significance ? 'S' : '<span style="color:#e94560">S</span>',
    ].join('');
    const qClass = q.pass ? 'ok' : 'fail';
    const rowCls = q.pass ? '' : ' failed';
    return `<tr onclick="selectPeak(${{p._idx}})" data-idx="${{p._idx}}" class="${{rowCls}}">
      <td>${{p.name}}</td>
      <td>${{p.energy}}</td>
      <td>${{p.significance}}</td>
      <td>${{ratStr}}</td>
      <td>${{p.chi2_red}}</td>
      <td class="${{qClass}}">${{qIcons}} (${{q.n_fail || 0}})</td>
      <td>${{act}}</td>
    </tr>`;
  }}).join('');

  document.querySelectorAll('#tableBody tr').forEach(tr => {{
    tr.classList.toggle('active', parseInt(tr.dataset.idx) === currentIdx);
  }});

  document.querySelectorAll('#summaryTable th').forEach(th => {{
    const col = th.dataset.col;
    th.querySelector('.arrow').textContent = col === sortCol ? (sortAsc ? '▲' : '▼') : '';
  }});
}}

document.querySelectorAll('#summaryTable th').forEach(th => {{
  th.addEventListener('click', () => {{
    const col = th.dataset.col;
    if (sortCol === col) sortAsc = !sortAsc;
    else {{ sortCol = col; sortAsc = true; }}
    renderTable();
  }});
}});

function selectPeak(idx) {{
  currentIdx = idx;
  const p = PEAKS[idx];
  renderTable();

  const row = document.querySelector(`#tableBody tr[data-idx="${{idx}}"]`);
  if (row) row.scrollIntoView({{ block: 'nearest' }});

  const traces = [
    {{
      x: [p.side_left[0], p.side_left[0], p.side_left[1], p.side_left[1]],
      y: [0, Math.max(...p.y_spec) * 1.1, Math.max(...p.y_spec) * 1.1, 0],
      fill: 'toself', fillcolor: 'rgba(255,165,0,0.1)', line: {{ width: 0 }},
      name: 'Sidebands', hoverinfo: 'skip', showlegend: true,
    }},
    {{
      x: [p.side_right[0], p.side_right[0], p.side_right[1], p.side_right[1]],
      y: [0, Math.max(...p.y_spec) * 1.1, Math.max(...p.y_spec) * 1.1, 0],
      fill: 'toself', fillcolor: 'rgba(255,165,0,0.1)', line: {{ width: 0 }},
      hoverinfo: 'skip', showlegend: false,
    }},
    {{
      x: [p.int_lims[0], p.int_lims[0], p.int_lims[1], p.int_lims[1]],
      y: [0, Math.max(...p.y_spec) * 1.1, Math.max(...p.y_spec) * 1.1, 0],
      fill: 'toself', fillcolor: 'rgba(233,69,96,0.08)', line: {{ width: 0 }},
      name: 'Integration', hoverinfo: 'skip', showlegend: true,
    }},
    {{
      x: p.x_fit, y: p.y_bkg,
      mode: 'lines', name: 'Background',
      line: {{ color: '#ff9800', width: 1.5, dash: 'dash' }},
    }},
    {{
      x: p.x_fit, y: p.y_fit,
      mode: 'lines', name: 'Fit (Gauss)',
      line: {{ color: '#e94560', width: 2 }},
    }},
    {{
      x: p.x_spec, y: p.y_spec,
      mode: 'markers', name: 'Data',
      marker: {{ color: '#4fc3f7', size: 4 }},
    }},
  ];

  const layout = {{
    margin: {{ t: 40, r: 30, b: 50, l: 60 }},
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
    font: {{ color: '#e0e0e0' }},
    xaxis: {{ title: 'Energy (keV)', gridcolor: '#2a2a4e', zeroline: false }},
    yaxis: {{ title: 'Counts', gridcolor: '#2a2a4e', zeroline: false }},
    title: {{ text: `${{p.name}}  —  ${{p.energy}} keV`, font: {{ size: 14, color: '#e94560' }} }},
    legend: {{ x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0.3)', font: {{ size: 10 }} }},
    shapes: [{{
      type: 'line', x0: p.energy, x1: p.energy, y0: 0, y1: Math.max(...p.y_spec) * 1.05,
      line: {{ color: '#4ecca3', width: 1, dash: 'dot' }},
    }}],
  }};

  Plotly.react('peakPlot', traces, layout, {{ responsive: true, displayModeBar: true }});

  const ratioClass = (p.ratio > 1.5 || p.ratio < 0.5) ? 'fail' : 'ok';
  const sigClass = p.significance >= 3 ? 'ok' : 'fail';
  const q = p.quality || {{}};
  const qChi2 = q.chi2 ? 'ok' : 'fail';
  const qSigma = q.sigma ? 'ok' : 'fail';
  const qRatio = q.ratio ? 'ok' : 'fail';
  const qSig = q.significance ? 'ok' : 'fail';
  const overallClass = q.pass ? 'ok' : 'fail';
  document.getElementById('infoBar').innerHTML = `
    <div class="item"><span class="label">Net Area (fit):</span><span class="value">${{p.net_area}}</span></div>
    <div class="item"><span class="label">Net Area (count):</span><span class="value">${{p.counting_net}}</span></div>
    <div class="item"><span class="label">Fit/Count:</span><span class="value ${{ratioClass}}">${{p.ratio}}</span></div>
    <div class="item"><span class="label">Significance:</span><span class="value ${{sigClass}}">${{p.significance}}σ</span></div>
    <div class="item"><span class="label">χ²/ndf:</span><span class="value ${{qChi2}}">${{p.chi2_red}}</span></div>
    <div class="item"><span class="label">σ (keV):</span><span class="value ${{qSigma}}">${{p.sigma}}${{p.sigma_at_bound ? ' *' : ''}}</span></div>
    <div class="item"><span class="label">Quality:</span><span class="value ${{overallClass}}">χ${{q.chi2 ? '✓' : '✗'}} σ${{q.sigma ? '✓' : '✗'}} R${{q.ratio ? '✓' : '✗'}} S${{q.significance ? '✓' : '✗'}} (${{q.n_fail || 0}} fail)</span></div>
    <div class="item"><span class="label">Activity:</span><span class="value">${{p.activity.toExponential(2)}} ± ${{p.activity_err.toExponential(2)}} Bq</span></div>
    <div class="item"><span class="label">Dose:</span><span class="value">${{p.dose.toExponential(2)}} µSv/h</span></div>
  `;

  document.getElementById('navCounter').textContent = `${{idx + 1}} / ${{PEAKS.length}}`;
}}

function navigate(dir) {{
  let next = currentIdx + dir;
  if (next < 0) next = PEAKS.length - 1;
  if (next >= PEAKS.length) next = 0;
  selectPeak(next);
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') navigate(-1);
  if (e.key === 'ArrowRight') navigate(1);
}});

init();
</script>
</body>
</html>"""
