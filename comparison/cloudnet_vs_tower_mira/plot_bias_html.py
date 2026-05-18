#!/usr/bin/env python3
"""
plot_bias_html.py
=================
Interactive HTML bias analysis for the cloudnet_vs_tower_mira workflow.

Reads the paired CSV files saved by compare_ze_fwd_both.py:
  output/comparison/cloudnet_vs_tower_mira/all/cn_paired_<period>.csv
  output/comparison/cloudnet_vs_tower_mira/all/tw_paired_<period>.csv

Produces:
  output/comparison/cloudnet_vs_tower_mira/all/bias_analysis_<period>.html

Layout:
  1. Stats table  (N, mean bias, median bias, 95% bootstrap CI, RMSE, slope, r)
  2. Violin plots of bias distribution per gate (CN and Tower)
  3. Scatter plots  Ze_FWD vs Ze_radar  (CN gate 0  |  Tower gate 6)
  4. Bias profile   bias ± CI vs height
  5. Monthly median bias evolution

Usage:
  python3 plot_bias_html.py                     # auto-finds latest CSVs
  python3 plot_bias_html.py 202502-202507       # explicit period label
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_BASE = (Path(__file__).resolve().parents[2] /
            'output' / 'comparison' / 'cloudnet_vs_tower_mira' / 'all')

TOWER_GATE = 6   # hardcoded — same as compare_ze_fwd_both.py

N_BOOT = 5000    # bootstrap resamples for CI


# ── Statistics helpers ────────────────────────────────────────────────────────

def bootstrap_ci(arr, statfn=np.nanmean, n=N_BOOT, ci=95):
    """Bootstrap confidence interval for statfn on 1-D arr."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan
    rng  = np.random.default_rng(42)
    boot = rng.choice(arr, size=(n, len(arr)), replace=True)
    dist = statfn(boot, axis=1)
    lo   = np.percentile(dist, (100 - ci) / 2)
    hi   = np.percentile(dist, 100 - (100 - ci) / 2)
    return lo, hi


def compute_stats(ze_fwd, ze_rad):
    """
    Full bias statistics for one (ze_fwd, ze_rad) pair of 1-D arrays (dBZ).
    Returns dict.
    """
    mask   = np.isfinite(ze_fwd) & np.isfinite(ze_rad)
    xf, xr = ze_fwd[mask], ze_rad[mask]
    N = len(xf)
    if N < 5:
        return dict(N=N, mean=np.nan, median=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    rmse=np.nan, std=np.nan, slope=np.nan, intercept=np.nan,
                    r=np.nan, p=np.nan)

    diff   = xr - xf
    mean_b = float(np.nanmean(diff))
    med_b  = float(np.nanmedian(diff))
    std_b  = float(np.nanstd(diff))
    rmse   = float(np.sqrt(np.nanmean(diff**2)))
    ci_lo, ci_hi = bootstrap_ci(diff)

    lr     = stats.linregress(xf, xr)
    r, p   = stats.pearsonr(xf, xr)

    return dict(N=N, mean=mean_b, median=med_b, ci_lo=float(ci_lo),
                ci_hi=float(ci_hi), rmse=rmse, std=std_b,
                slope=float(lr.slope), intercept=float(lr.intercept),
                r=float(r), p=float(p))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(period_label):
    cn_path = OUT_BASE / f'cn_paired_{period_label}.csv'
    tw_path = OUT_BASE / f'tw_paired_{period_label}.csv'
    if not cn_path.exists() or not tw_path.exists():
        raise FileNotFoundError(
            f'Missing paired CSVs in {OUT_BASE}\n'
            f'  Expected: cn_paired_{period_label}.csv  and  tw_paired_{period_label}.csv\n'
            f'  Run: compare_ze_fwd_both.py 202502 202503 ... 202507  first.')
    cn = pd.read_csv(cn_path)
    tw = pd.read_csv(tw_path)
    return cn, tw


def get_cn_gate_cols(cn_df):
    """Return list of Ze column names (height-label gate columns) in Cloudnet DF."""
    return [c for c in cn_df.columns if c not in ('month', 'ze_fwd', 'rr')]


def get_tw_gate_cols(tw_df):
    """Return list of Ze column names for Tower gates."""
    return [c for c in tw_df.columns if c.startswith('ze_gate')]


def parse_height(col):
    """Extract numeric height in metres from a column name like '255m' or 'ze_gate6_231m'."""
    m = re.search(r'(\d+)m', col)
    return int(m.group(1)) if m else 0


# ── Plot builders ─────────────────────────────────────────────────────────────

MONTH_LABELS = {
    '202502': 'Feb 25', '202503': 'Mar 25', '202504': 'Apr 25',
    '202505': 'May 25', '202506': 'Jun 25', '202507': 'Jul 25',
}

CN_COLOUR  = '#1f77b4'
TW_COLOUR  = '#d62728'
ZERO_STYLE = dict(line_color='black', line_width=1, line_dash='dash', opacity=0.4)


def _violin_traces(df, gate_cols, colour, name_prefix, show_legend=True):
    """One go.Violin per gate column, showing bias = Ze_radar - Ze_fwd."""
    traces = []
    for col in gate_cols:
        bias = df[col].values - df['ze_fwd'].values
        bias = bias[np.isfinite(bias)]
        h    = parse_height(col)
        traces.append(go.Violin(
            y=bias,
            name=f'{h} m',
            legendgroup=name_prefix,
            legendgrouptitle_text=name_prefix if col == gate_cols[0] else None,
            showlegend=show_legend and (col == gate_cols[0]),
            box_visible=True,
            meanline_visible=True,
            fillcolor=colour,
            opacity=0.7,
            line_color=colour,
            points=False,
            hovertemplate=(
                f'<b>{name_prefix} {h} m</b><br>'
                'Bias: %{y:.2f} dBZ<extra></extra>'
            ),
        ))
    return traces


def build_violin_fig(cn_df, tw_df, cn_cols, tw_cols, period):
    """Two-panel violin figure: left=Cloudnet gates, right=Tower gates."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Cloudnet  (Ze_radar − Ze_FWD)  per gate',
                        'Tower ZNC  (Ze_radar − Ze_FWD)  per gate'],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    for tr in _violin_traces(cn_df, cn_cols, CN_COLOUR, 'Cloudnet'):
        fig.add_trace(tr, row=1, col=1)
    for tr in _violin_traces(tw_df, tw_cols, TW_COLOUR, 'Tower ZNC', show_legend=False):
        fig.add_trace(tr, row=1, col=2)

    fig.add_hline(y=0, row=1, col=1, **ZERO_STYLE)
    fig.add_hline(y=0, row=1, col=2, **ZERO_STYLE)
    fig.update_yaxes(title_text='Bias  Ze_radar − Ze_FWD  [dB]', row=1, col=1)
    fig.update_xaxes(title_text='Height ASL [m]')
    fig.update_layout(
        title=f'Bias distributions per gate  |  {period}',
        height=500, width=1300,
        violinmode='group',
        plot_bgcolor='white',
        margin=dict(t=70, b=50),
    )
    return fig


def build_scatter_fig(cn_df, tw_df, cn_cols, tw_cols, period):
    """Two-panel scatter: CN gate 0 | Tower gate TOWER_GATE."""
    cn_col = cn_cols[0]
    tw_col = tw_cols[TOWER_GATE] if TOWER_GATE < len(tw_cols) else tw_cols[-1]
    cn_h   = parse_height(cn_col)
    tw_h   = parse_height(tw_col)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'Cloudnet  {cn_h} m ASL',
            f'Tower ZNC  {tw_h} m ASL  (gate {TOWER_GATE})',
        ],
        horizontal_spacing=0.10,
    )

    lim = (-10, 55)

    for col_i, (df, zcol, label, colour) in enumerate([
        (cn_df, cn_col, 'Cloudnet', CN_COLOUR),
        (tw_df, tw_col, 'Tower ZNC', TW_COLOUR),
    ], start=1):
        mask = np.isfinite(df['ze_fwd'].values) & np.isfinite(df[zcol].values)
        xf   = df['ze_fwd'].values[mask]
        xr   = df[zcol].values[mask]
        rr   = df['rr'].values[mask]
        mon  = df['month'].values[mask].astype(str)

        st   = compute_stats(xf, xr)
        reg_x = np.array(lim)
        reg_y = st['slope'] * reg_x + st['intercept']

        fig.add_trace(go.Scatter(
            x=xf, y=xr, mode='markers',
            marker=dict(size=4, color=rr, colorscale='Plasma',
                        cmin=0, cmax=10, opacity=0.6,
                        colorbar=dict(title='RR [mm/h]', thickness=12,
                                      x=1.01 if col_i == 2 else -0.12,
                                      len=0.6) if col_i == 2 else None),
            text=[f'Month: {MONTH_LABELS.get(m, m)}<br>RR: {r:.1f} mm/h'
                  for m, r in zip(mon, rr)],
            hovertemplate='Ze_FWD: %{x:.1f} dBZ<br>Ze_radar: %{y:.1f} dBZ<br>%{text}<extra></extra>',
            name=label, showlegend=False,
        ), row=1, col=col_i)

        # 1:1 line
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode='lines',
            line=dict(color='black', dash='dash', width=1),
            showlegend=(col_i == 1), name='1:1',
        ), row=1, col=col_i)

        # regression line
        fig.add_trace(go.Scatter(
            x=reg_x.tolist(), y=reg_y.tolist(), mode='lines',
            line=dict(color=colour, width=2),
            showlegend=(col_i == 1),
            name=f'Fit  (slope={st["slope"]:.2f}, b={st["intercept"]:.1f} dB)',
        ), row=1, col=col_i)

        ann = (f'N={st["N"]}  bias={st["mean"]:+.2f} dB<br>'
               f'median={st["median"]:+.2f} dB  RMSE={st["rmse"]:.2f} dB<br>'
               f'95% CI [{st["ci_lo"]:+.2f}, {st["ci_hi"]:+.2f}]  r={st["r"]:.3f}')
        fig.add_annotation(
            xref='x domain' if col_i == 1 else 'x2 domain',
            yref='y domain' if col_i == 1 else 'y2 domain',
            x=0.03, y=0.97, xanchor='left', yanchor='top',
            text=ann, showarrow=False,
            bgcolor='rgba(255,255,255,0.85)', bordercolor=colour,
            font=dict(size=10), align='left',
        )

    fig.update_xaxes(range=lim, title_text='Ze FWD [dBZ]')
    fig.update_yaxes(range=lim, title_text='Ze radar [dBZ]', col=1)
    fig.update_layout(
        title=f'Scatter: Ze FWD vs Ze radar  |  {period}',
        height=520, width=1200,
        plot_bgcolor='white',
        margin=dict(t=70),
    )
    return fig


def build_profile_fig(cn_df, tw_df, cn_cols, tw_cols, period):
    """Horizontal bias profile: bias ± 95% bootstrap CI vs height."""
    fig = go.Figure()

    for df, cols, label, colour, sym in [
        (cn_df, cn_cols, 'Cloudnet', CN_COLOUR, 'circle'),
        (tw_df, tw_cols, 'Tower ZNC', TW_COLOUR, 'diamond'),
    ]:
        heights, means, lo, hi, meds, Ns = [], [], [], [], [], []
        for col in cols:
            st = compute_stats(df['ze_fwd'].values, df[col].values)
            if st['N'] < 5:
                continue
            heights.append(parse_height(col))
            means.append(st['mean'])
            meds.append(st['median'])
            lo.append(st['ci_lo'])
            hi.append(st['ci_hi'])
            Ns.append(st['N'])

        heights = np.array(heights)

        fig.add_trace(go.Scatter(
            x=means, y=heights, mode='lines+markers',
            name=f'{label}  mean',
            marker=dict(size=8, symbol=sym, color=colour),
            line=dict(color=colour, width=2),
            error_x=dict(
                type='data',
                arrayminus=[m - l for m, l in zip(means, lo)],
                array=[h - m for h, m in zip(hi, means)],
                color=colour, thickness=1.5, width=5,
            ),
            hovertemplate=(
                f'<b>{label}</b><br>Height: %{{y}} m<br>'
                'Mean bias: %{x:.2f} dB<br>%{text}<extra></extra>'
            ),
            text=[f'N={n}  95% CI [{l:+.2f}, {h:+.2f}]'
                  for n, l, h in zip(Ns, lo, hi)],
        ))

        fig.add_trace(go.Scatter(
            x=meds, y=heights, mode='markers',
            name=f'{label}  median',
            marker=dict(size=6, symbol=sym + '-open', color=colour),
            hovertemplate=f'<b>{label} median</b><br>Height: %{{y}} m<br>Bias: %{{x:.2f}} dB<extra></extra>',
        ))

    fig.add_vline(x=0, **ZERO_STYLE)
    fig.update_xaxes(title_text='Bias  Ze_radar − Ze_FWD  [dB]')
    fig.update_yaxes(title_text='Height ASL [m]')
    fig.update_layout(
        title=f'Bias profile ± 95% bootstrap CI  |  {period}',
        height=500, width=700,
        plot_bgcolor='white',
        margin=dict(t=70),
    )
    return fig


def build_monthly_fig(cn_df, tw_df, period):
    """Monthly median bias for CN gate 0 and Tower gate TOWER_GATE."""
    months = sorted(cn_df['month'].astype(str).unique())

    cn_col = [c for c in cn_df.columns if c not in ('month', 'ze_fwd', 'rr')][0]
    tw_cols_all = [c for c in tw_df.columns if c.startswith('ze_gate')]
    tw_col = tw_cols_all[TOWER_GATE] if TOWER_GATE < len(tw_cols_all) else tw_cols_all[-1]
    cn_h = parse_height(cn_col)
    tw_h = parse_height(tw_col)

    fig = go.Figure()

    for df, zcol, label, colour, sym in [
        (cn_df, cn_col, f'Cloudnet {cn_h} m', CN_COLOUR, 'circle'),
        (tw_df, tw_col, f'Tower {tw_h} m', TW_COLOUR, 'diamond'),
    ]:
        meds, means, lo, hi, Ns, xlabels = [], [], [], [], [], []
        for ym in months:
            sub = df[df['month'].astype(str) == ym]
            bias = sub[zcol].values - sub['ze_fwd'].values
            bias = bias[np.isfinite(bias)]
            if len(bias) < 5:
                continue
            xlabels.append(MONTH_LABELS.get(ym, ym))
            meds.append(float(np.median(bias)))
            means.append(float(np.mean(bias)))
            c_lo, c_hi = bootstrap_ci(bias)
            lo.append(float(c_lo))
            hi.append(float(c_hi))
            Ns.append(len(bias))

        fig.add_trace(go.Scatter(
            x=xlabels, y=means, mode='lines+markers',
            name=f'{label}  mean',
            marker=dict(size=9, symbol=sym, color=colour),
            line=dict(color=colour, width=2),
            error_y=dict(
                type='data',
                arrayminus=[m - l for m, l in zip(means, lo)],
                array=[h - m for h, m in zip(hi, means)],
                color=colour, thickness=1.5, width=6,
            ),
            hovertemplate='%{x}<br>Mean bias: %{y:.2f} dB<br>%{text}<extra></extra>',
            text=[f'N={n}  95% CI [{l:+.2f}, {h:+.2f}]'
                  for n, l, h in zip(Ns, lo, hi)],
        ))
        fig.add_trace(go.Scatter(
            x=xlabels, y=meds, mode='markers',
            name=f'{label}  median',
            marker=dict(size=7, symbol=sym + '-open', color=colour),
            hovertemplate='%{x}<br>Median bias: %{y:.2f} dB<extra></extra>',
        ))

    fig.add_hline(y=0, **ZERO_STYLE)
    fig.update_yaxes(title_text='Bias  Ze_radar − Ze_FWD  [dB]')
    fig.update_layout(
        title=f'Monthly bias evolution (mean ± 95% CI, median)  |  {period}',
        height=420, width=850,
        plot_bgcolor='white',
        margin=dict(t=70),
    )
    return fig


def build_stats_table(cn_df, tw_df, cn_cols, tw_cols, period):
    """Plotly table with per-gate statistics for CN and Tower."""
    rows_cn, rows_tw = [], []

    for col in cn_cols:
        st = compute_stats(cn_df['ze_fwd'].values, cn_df[col].values)
        h  = parse_height(col)
        rows_cn.append([
            f'CN  {h} m',
            st['N'],
            f'{st["mean"]:+.2f}',
            f'{st["median"]:+.2f}',
            f'[{st["ci_lo"]:+.2f}, {st["ci_hi"]:+.2f}]',
            f'{st["rmse"]:.2f}',
            f'{st["std"]:.2f}',
            f'{st["slope"]:.3f}',
            f'{st["intercept"]:+.2f}',
            f'{st["r"]:.3f}',
        ])

    for g, col in enumerate(tw_cols):
        st = compute_stats(tw_df['ze_fwd'].values, tw_df[col].values)
        h  = parse_height(col)
        marker = '  ←' if g == TOWER_GATE else ''
        rows_tw.append([
            f'TW gate{g}  {h} m{marker}',
            st['N'],
            f'{st["mean"]:+.2f}',
            f'{st["median"]:+.2f}',
            f'[{st["ci_lo"]:+.2f}, {st["ci_hi"]:+.2f}]',
            f'{st["rmse"]:.2f}',
            f'{st["std"]:.2f}',
            f'{st["slope"]:.3f}',
            f'{st["intercept"]:+.2f}',
            f'{st["r"]:.3f}',
        ])

    all_rows = rows_cn + rows_tw
    headers  = ['Sensor / Gate', 'N', 'Mean bias [dB]', 'Median bias [dB]',
                '95% boot CI', 'RMSE [dB]', 'Std [dB]', 'Slope', 'Intercept [dB]', 'Pearson r']
    n_cn = len(rows_cn)

    cell_colors = [
        ['#dce9f5'] * n_cn + ['#fde0de'] * len(rows_tw),  # row colours per gate
    ] + [['white'] * len(all_rows)] * (len(headers) - 1)

    fig = go.Figure(go.Table(
        header=dict(values=headers,
                    fill_color='#404040', font_color='white',
                    align='center', font_size=12, height=28),
        cells=dict(
            values=[[r[i] for r in all_rows] for i in range(len(headers))],
            fill_color=cell_colors,
            align=['left'] + ['center'] * (len(headers) - 1),
            font_size=11, height=24,
        ),
    ))
    fig.update_layout(
        title=f'Bias statistics per gate  |  {period}  (bootstrap CI: {N_BOOT} resamples)',
        height=60 + 28 + 24 * len(all_rows) + 40,
        margin=dict(t=60, b=20),
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def make_html(period_label):
    print(f'Loading paired data for period: {period_label} …')
    cn_df, tw_df = load_data(period_label)

    cn_cols = get_cn_gate_cols(cn_df)
    tw_cols = get_tw_gate_cols(tw_df)

    print(f'  CN: {len(cn_df)} rows  |  Tower: {len(tw_df)} rows')
    print(f'  CN gates: {[parse_height(c) for c in cn_cols]} m ASL')
    print(f'  TW gates: {[parse_height(c) for c in tw_cols]} m ASL')
    print('Computing statistics and building HTML …')

    figs = [
        ('stats_table',  build_stats_table(cn_df, tw_df, cn_cols, tw_cols, period_label)),
        ('violins',      build_violin_fig(cn_df, tw_df, cn_cols, tw_cols, period_label)),
        ('scatter',      build_scatter_fig(cn_df, tw_df, cn_cols, tw_cols, period_label)),
        ('profile',      build_profile_fig(cn_df, tw_df, cn_cols, tw_cols, period_label)),
        ('monthly',      build_monthly_fig(cn_df, tw_df, period_label)),
    ]

    # Combine into one HTML
    html_parts = [
        '<!DOCTYPE html><html><head>',
        '<meta charset="utf-8">',
        f'<title>Ze Bias Analysis — {period_label}</title>',
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
        '<style>body{font-family:sans-serif;background:#f5f5f5;padding:20px}'
        'h1{color:#222;font-size:1.3em}'
        '.section{background:white;border-radius:8px;padding:16px;'
        'margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.12)}'
        '.section h2{margin-top:0;font-size:1.05em;color:#444}'
        '</style></head><body>',
        f'<h1>Ze Bias Analysis: FWD vs Radar &nbsp;|&nbsp; {period_label}</h1>',
        '<p style="color:#666;font-size:.9em">'
        'Bias = Ze_radar − Ze_FWD (dBZ). Negative = radar reads lower than FWD simulation. '
        f'Bootstrap CI uses {N_BOOT} resamples. '
        'Tower gate used: gate 6 (~231 m ASL).</p>',
    ]

    section_titles = {
        'stats_table': 'Statistics Table',
        'violins':     'Bias Distributions (Violin Plots)',
        'scatter':     'Scatter: Ze FWD vs Ze Radar',
        'profile':     'Bias Profile vs Height',
        'monthly':     'Monthly Bias Evolution',
    }

    for name, fig in figs:
        div_html = fig.to_html(full_html=False, include_plotlyjs=False,
                               config={'scrollZoom': True, 'displayModeBar': True})
        html_parts += [
            f'<div class="section">',
            f'<h2>{section_titles[name]}</h2>',
            div_html,
            '</div>',
        ]

    html_parts.append('</body></html>')

    outfile = OUT_BASE / f'bias_analysis_{period_label}.html'
    outfile.write_text('\n'.join(html_parts), encoding='utf-8')
    print(f'Saved → {outfile}')
    print(f'Open:   file://{outfile.resolve()}')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        period = sys.argv[1]
    else:
        # auto-find latest CSV
        csvs = sorted(OUT_BASE.glob('cn_paired_*.csv'))
        if not csvs:
            print(f'No cn_paired_*.csv found in {OUT_BASE}')
            print('Run compare_ze_fwd_both.py first.')
            sys.exit(1)
        period = csvs[-1].stem.replace('cn_paired_', '')
        print(f'Auto-detected period: {period}')

    make_html(period)
