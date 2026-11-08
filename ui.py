"""HTML building blocks for the Streamlit app: insight card and dashboard panels."""
import html

import numpy as np
import pandas as pd

GREEN = "#4ade80"
ORANGE = "#fb923c"
RED = "#f87171"
ACTION_COLOR = {"Fast-track": GREEN, "Manual review": "#facc15", "Reject or require collateral": RED}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, button, input { font-family: 'Inter', sans-serif; }
.stApp { background: #0a0d0b; }
.block-container { padding-top: 2rem; max-width: 1280px; }
.cr-hero { border-radius: 24px; padding: 34px 36px 30px; margin-bottom: 18px;
  background: radial-gradient(120% 140% at 85% 0%, #1f7a3a 0%, #0f2a17 38%, #0b0f0c 75%);
  border: 1px solid #1c2a20; }
.cr-hero small { color: #9fb3a5; font-size: 14px; }
.cr-hero h1 { color: #f4f7f4; font-size: 40px; font-weight: 600; margin: 6px 0 6px; letter-spacing: -0.5px; }
.cr-hero p { color: #b9c7bd; margin: 0; font-size: 15px; max-width: 720px; }
.cr-card { background: #121714; border: 1px solid #1f2722; border-radius: 20px; padding: 20px 22px; height: 100%; }
.cr-card h4 { color: #e8ece9; font-size: 15px; font-weight: 600; margin: 0 0 14px; }
.cr-muted { color: #8b9a90; font-size: 13px; }
.cr-kpi-value { color: #f4f7f4; font-size: 32px; font-weight: 600; margin: 6px 0 2px; letter-spacing: -0.5px; }
.cr-row { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.cr-seg { display: flex; gap: 3px; height: 10px; margin-top: 14px; }
.cr-seg div { border-radius: 3px; }
.cr-bars { display: flex; align-items: flex-end; gap: 4px; height: 170px; padding-top: 8px; }
.cr-bars .b { flex: 1; border-radius: 4px 4px 0 0; min-height: 2px; }
.cr-axis { display: flex; justify-content: space-between; color: #6f7d74; font-size: 12px; margin-top: 6px; }
.cr-act { margin-bottom: 14px; }
.cr-act .track { height: 8px; background: #1e2621; border-radius: 4px; margin-top: 6px; overflow: hidden; }
.cr-act .fill { height: 8px; border-radius: 4px; }
.cr-insight { position: relative; border-radius: 28px; padding: 26px 26px 24px; overflow: hidden; min-height: 380px;
  background: radial-gradient(90% 70% at 0% 40%, rgba(251,146,60,.55) 0%, rgba(251,146,60,0) 60%),
              radial-gradient(120% 90% at 30% 0%, #1f7a3a 0%, #0f2416 45%, #090b0a 80%);
  border: 1px solid #1c2a20; color: #f4f7f4; display: flex; flex-direction: column; }
.cr-insight .top { display: flex; justify-content: space-between; align-items: center; }
.cr-pill { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; color: #e8ece9; }
.cr-pill .ic { width: 22px; height: 22px; border-radius: 6px; background: #f4f7f4; display: inline-block; }
.cr-chip { border: 1px solid rgba(255,255,255,.35); border-radius: 999px; padding: 5px 12px; font-size: 13px; }
.cr-big { font-size: 84px; font-weight: 300; letter-spacing: -2px; line-height: 1; margin: 64px 0 0; }
.cr-big sup { font-size: 30px; font-weight: 400; margin-left: 6px; }
.cr-lead { font-size: 21px; line-height: 1.35; margin: 56px 0 10px; }
.cr-lead span { color: #8b9a90; }
.cr-sub { color: #a7b4ab; font-size: 13.5px; line-height: 1.5; }
.cr-dots { display: flex; gap: 6px; margin-top: auto; padding-top: 26px; }
.cr-dots div { flex: 1; height: 4px; border-radius: 2px; background: rgba(255,255,255,.22); }
.cr-dots div.on { background: #f4f7f4; }
.cr-guide div.item { padding: 12px 14px; border-radius: 12px; margin-bottom: 8px; border: 1px solid #1f2722; }
.cr-guide div.item.on { border-color: #4ade80; background: rgba(74,222,128,.07); }
.cr-guide b { color: #e8ece9; font-weight: 600; }
</style>
"""


def esc(text):
    return html.escape(str(text))


def insight_card(score, band, action, band_bad_rate):
    arrow, color = ("&#8599;", ORANGE) if score >= 0.2 else ("&#8600;", GREEN)
    dots = "".join(f'<div class="{"on" if i == band else ""}"></div>' for i in range(1, 11))
    return f"""
<div class="cr-insight">
  <div class="top">
    <span class="cr-pill"><span class="ic"></span>Risk insight</span>
    <span class="cr-chip">Band {band} of 10</span>
  </div>
  <div class="cr-big">{score * 100:.1f}%<sup style="color:{color}">{arrow}</sup></div>
  <div class="cr-lead">probability of default. <span>Recommended: {esc(action)}.</span></div>
  <div class="cr-sub">In the 2014 test data, {band_bad_rate:.1%} of loans in risk band {band} ended as bad loans.</div>
  <div class="cr-dots">{dots}</div>
</div>"""


def decision_guide(score, review_cutoff, reject_cutoff, action):
    items = [("Fast-track", f"Probability below {review_cutoff:.2f}"),
             ("Manual review", f"{review_cutoff:.2f} to {reject_cutoff:.2f}"),
             ("Reject or require collateral", f"{reject_cutoff:.2f} and above")]
    rows = "".join(
        f'<div class="item {"on" if name == action else ""}"><b>{name}</b><div class="cr-muted">{rule}</div></div>'
        for name, rule in items)
    return f'<div class="cr-card cr-guide"><h4>Decision guide</h4>{rows}' \
           f'<div class="cr-muted" style="margin-top:12px">The score supports a decision and does not replace ' \
           f'human review.</div></div>'


def hero(title, subtitle, eyebrow):
    return f'<div class="cr-hero"><small>{esc(eyebrow)}</small><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>'


def kpi(title, value, note_left, note_right, share):
    share = float(np.clip(share, 0, 1))
    n_on = int(round(share * 24))
    seg = "".join(f'<div style="flex:1;background:{GREEN if i < n_on else "#26302a"}"></div>' for i in range(24))
    return f"""<div class="cr-card"><div class="cr-muted">{esc(title)}</div>
<div class="cr-kpi-value">{esc(value)}</div>
<div class="cr-row cr-muted"><span>{esc(note_left)}</span><span>{esc(note_right)}</span></div>
<div class="cr-seg">{seg}</div></div>"""


def histogram(scores, review_cutoff, reject_cutoff, bins=24):
    top = max(0.8, float(np.quantile(scores, 0.995)))
    counts, edges = np.histogram(np.clip(scores, 0, top), bins=bins, range=(0, top))
    peak = counts.max() if counts.max() else 1
    bars = []
    for c, lo in zip(counts, edges[:-1]):
        color = GREEN if lo < review_cutoff else ("#facc15" if lo < reject_cutoff else RED)
        bars.append(f'<div class="b" title="{c} applications" style="height:{c / peak * 100:.1f}%;background:{color}"></div>')
    return f"""<div class="cr-card"><h4>Distribution of probability of default</h4>
<div class="cr-bars">{''.join(bars)}</div>
<div class="cr-axis"><span>0%</span><span>{top / 2:.0%}</span><span>{top:.0%}</span></div>
<div class="cr-muted" style="margin-top:10px">Green: fast-track, yellow: manual review, red: reject or require collateral.</div></div>"""


def action_breakdown(actions):
    total = len(actions)
    rows = []
    for name, color in ACTION_COLOR.items():
        n = int((actions == name).sum())
        rows.append(f"""<div class="cr-act"><div class="cr-row"><span style="color:#e8ece9">{name}</span>
<span class="cr-muted">{n:,} &middot; {n / total:.0%}</span></div>
<div class="track"><div class="fill" style="width:{n / total * 100:.1f}%;background:{color}"></div></div></div>""")
    return f'<div class="cr-card"><h4>Recommended actions</h4>{"".join(rows)}</div>'


def band_chart(bands, actual=None):
    """Applications per risk band; with outcomes, bar height shows the actual bad rate instead."""
    data = pd.DataFrame({"band": bands})
    if actual is not None:
        data["bad"] = actual
        stats = data.groupby("band")["bad"].agg(["size", "mean"]).reindex(range(1, 11))
        values, title = stats["mean"], "Actual bad rate by risk band"
        labels = [f"{v:.0%}" if pd.notna(v) else "" for v in values]
        peak = max(values.max(), 0.01) if values.notna().any() else 1
    else:
        values = data["band"].value_counts().reindex(range(1, 11), fill_value=0)
        title, labels, peak = "Applications by risk band", [f"{v:,}" for v in values], max(values.max(), 1)
    bars = "".join(
        f'<div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;height:100%;text-align:center">'
        f'<div class="cr-muted" style="font-size:11px;margin-bottom:4px">{lab}</div>'
        f'<div style="height:{(0 if pd.isna(v) else v) / peak * 82:.1f}%;min-height:2px;border-radius:4px 4px 0 0;'
        f'background:linear-gradient(180deg,{GREEN},#166534)"></div></div>'
        for v, lab in zip(values, labels))
    axis = "".join(f'<span style="flex:1;text-align:center">{i}</span>' for i in range(1, 11))
    return f"""<div class="cr-card"><h4>{title}</h4>
<div style="display:flex;gap:6px;height:170px">{bars}</div>
<div class="cr-axis" style="gap:6px">{axis}</div>
<div class="cr-muted" style="margin-top:10px">Band 1 is the safest, band 10 the riskiest.</div></div>"""
