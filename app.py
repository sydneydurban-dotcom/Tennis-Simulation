import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="The Tennis Model",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Clean CSS — light theme, readable, sectioned
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

section[data-testid="stMainBlockContainer"] { max-width: 1100px; }

[data-testid="stSidebar"] {
    background: #f0f4f0;
    border-right: 2px solid #c8dcc8;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #1a3a1a !important; }

h1 {
    font-family: 'Playfair Display', serif !important;
    color: #1a3a1a !important;
    font-weight: 800 !important;
    font-size: 2.4rem !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
}
h2 {
    font-family: 'Playfair Display', serif !important;
    color: #1a3a1a !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
    border-bottom: 2px solid #2e7d32;
    padding-bottom: 8px;
    margin-top: 2rem !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    color: #2d5a2d !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
}
p, li, span, label, div {
    font-family: 'Inter', sans-serif !important;
    color: #2a2a2a !important;
}

[data-testid="stMetric"] {
    background: #f5faf5;
    border: 1px solid #c8dcc8;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetric"] label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #3d6b3d !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #1a3a1a !important;
    font-weight: 600 !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
}

[data-testid="stExpander"] {
    border: 1px solid #d4e4d4 !important;
    border-radius: 10px !important;
    background: #fafcfa;
    margin-bottom: 0.5rem;
}
[data-testid="stExpander"] summary span {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    color: #2d5a2d !important;
}

.phase-card {
    background: #f5faf5;
    border: 1px solid #c8dcc8;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 1rem;
}
.phase-card h4 {
    font-family: 'Playfair Display', serif !important;
    color: #1a3a1a !important;
    font-size: 1.2rem !important;
    margin: 0 0 8px 0;
}
.phase-card p {
    color: #4a6a4a !important;
    font-size: 0.92rem !important;
    margin-bottom: 12px;
}

.state-badge {
    display: inline-block;
    background: #e8f0e8;
    color: #2d5a2d;
    padding: 4px 12px;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px;
    border: 1px solid #c8dcc8;
}
.state-badge.duel {
    background: #fff3e0;
    color: #e65100;
    border-color: #ffcc80;
}
.state-badge.absorb {
    background: #1a3a1a;
    color: #c8e6c9;
    border-color: #1a3a1a;
}

.section-break {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #a5d6a7, transparent);
    margin: 2.5rem 0 1.5rem 0;
}

.subtitle {
    font-family: 'Inter', sans-serif;
    color: #5a7a5a;
    font-size: 1.05rem;
    font-weight: 500;
    margin-top: -8px;
    margin-bottom: 1.5rem;
}

.param-box {
    background: #1a3a1a;
    padding: 16px 20px;
    border-radius: 10px;
    margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    line-height: 1.8;
}
.param-box span { color: #a5d6a7 !important; }
.param-box .val { color: #ffffff !important; font-weight: 600; }

.info-callout {
    background: #e8f5e9;
    border-left: 4px solid #2e7d32;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 1rem 0;
    font-size: 0.92rem;
    color: #1a3a1a;
}

.stButton button[kind="primary"] {
    background: #2e7d32 !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ═══════════════════════════════════════════════════════════════

TRANSIENT_STATES = [
    "0-0", "1-0", "0-1", "2-0", "1-1", "0-2",
    "3-0", "2-1", "1-2", "0-3", "3-1", "1-3",
    "Deuce", "Ad-In", "Ad-Out",
]
ABSORBING_STATES = ["Server Wins", "Returner Wins"]
ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES
N_TRANSIENT = len(TRANSIENT_STATES)
N_ABSORBING = len(ABSORBING_STATES)
N_TOTAL = N_TRANSIENT + N_ABSORBING

TENNIS_SCORE_MAP = {
    "0-0": "Love-All", "1-0": "15-Love", "0-1": "Love-15",
    "2-0": "30-Love", "1-1": "15-All", "0-2": "Love-30",
    "3-0": "40-Love", "2-1": "30-15", "1-2": "15-30", "0-3": "Love-40",
    "3-1": "40-15", "1-3": "15-40",
    "Deuce": "Deuce (30-30+)", "Ad-In": "Ad Server", "Ad-Out": "Ad Returner",
    "Server Wins": "Game, Server", "Returner Wins": "Game, Returner",
}
SPRINT_STATES = ["0-0","1-0","0-1","2-0","1-1","0-2","3-0","2-1","1-2","0-3","3-1","1-3"]
DUEL_STATES = ["Deuce","Ad-In","Ad-Out"]


def build_transition_matrix(p):
    q = 1.0 - p
    P = np.zeros((N_TOTAL, N_TOTAL))
    idx = {name: i for i, name in enumerate(ALL_STATES)}
    P[idx["0-0"], idx["1-0"]] = p;   P[idx["0-0"], idx["0-1"]] = q
    P[idx["1-0"], idx["2-0"]] = p;   P[idx["1-0"], idx["1-1"]] = q
    P[idx["0-1"], idx["1-1"]] = p;   P[idx["0-1"], idx["0-2"]] = q
    P[idx["2-0"], idx["3-0"]] = p;   P[idx["2-0"], idx["2-1"]] = q
    P[idx["1-1"], idx["2-1"]] = p;   P[idx["1-1"], idx["1-2"]] = q
    P[idx["0-2"], idx["1-2"]] = p;   P[idx["0-2"], idx["0-3"]] = q
    P[idx["3-0"], idx["Server Wins"]] = p;   P[idx["3-0"], idx["3-1"]] = q
    P[idx["2-1"], idx["3-1"]] = p;           P[idx["2-1"], idx["Deuce"]] = q
    P[idx["1-2"], idx["Deuce"]] = p;         P[idx["1-2"], idx["1-3"]] = q
    P[idx["0-3"], idx["1-3"]] = p;           P[idx["0-3"], idx["Returner Wins"]] = q
    P[idx["3-1"], idx["Server Wins"]] = p;   P[idx["3-1"], idx["Ad-In"]] = q
    P[idx["1-3"], idx["Ad-Out"]] = p;        P[idx["1-3"], idx["Returner Wins"]] = q
    P[idx["Deuce"], idx["Ad-In"]] = p;       P[idx["Deuce"], idx["Ad-Out"]] = q
    P[idx["Ad-In"], idx["Server Wins"]] = p; P[idx["Ad-In"], idx["Deuce"]] = q
    P[idx["Ad-Out"], idx["Deuce"]] = p;      P[idx["Ad-Out"], idx["Returner Wins"]] = q
    P[idx["Server Wins"], idx["Server Wins"]] = 1.0
    P[idx["Returner Wins"], idx["Returner Wins"]] = 1.0
    return P


def analytical_engine(P):
    Q = P[:N_TRANSIENT, :N_TRANSIENT]
    R = P[:N_TRANSIENT, N_TRANSIENT:]
    F = np.linalg.inv(np.eye(N_TRANSIENT) - Q)
    B = F @ R
    return Q, R, F, B, F.sum(axis=1)


def monte_carlo_engine(p, n_trials):
    idx = {name: i for i, name in enumerate(ALL_STATES)}
    absorbing_set = {idx["Server Wins"], idx["Returner Wins"]}
    server_win_idx = idx["Server Wins"]
    mapping = {
        "0-0": ("1-0","0-1"), "1-0": ("2-0","1-1"), "0-1": ("1-1","0-2"),
        "2-0": ("3-0","2-1"), "1-1": ("2-1","1-2"), "0-2": ("1-2","0-3"),
        "3-0": ("Server Wins","3-1"), "2-1": ("3-1","Deuce"),
        "1-2": ("Deuce","1-3"), "0-3": ("1-3","Returner Wins"),
        "3-1": ("Server Wins","Ad-In"), "1-3": ("Ad-Out","Returner Wins"),
        "Deuce": ("Ad-In","Ad-Out"),
        "Ad-In": ("Server Wins","Deuce"), "Ad-Out": ("Deuce","Returner Wins"),
    }
    transitions = {idx[k]: (idx[v[0]], idx[v[1]]) for k, v in mapping.items()}

    server_wins = 0
    total_points = 0
    state_visits = np.zeros(N_TRANSIENT)
    point_counts = []
    win_sequence = []

    for _ in range(n_trials):
        state = 0
        points = 0
        while state not in absorbing_set:
            state_visits[state] += 1
            s_next, r_next = transitions[state]
            state = s_next if random.random() < p else r_next
            points += 1
        total_points += points
        point_counts.append(points)
        won = state == server_win_idx
        if won: server_wins += 1
        win_sequence.append(1 if won else 0)

    return {
        "server_win_prob": server_wins / n_trials,
        "returner_win_prob": 1.0 - server_wins / n_trials,
        "expected_points": total_points / n_trials,
        "state_visit_probs": state_visits / n_trials,
        "point_counts": point_counts,
        "win_sequence": win_sequence,
    }


# ─────────────────────────────────────────────
# Visualization helpers
# ─────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="#fafcfa",
    font=dict(family="Inter, sans-serif", color="#2a2a2a", size=12),
    margin=dict(l=50, r=30, t=40, b=50),
)
GRID = dict(gridcolor="#e0e8e0", gridwidth=1)


def make_state_diagram(p):
    positions = {
        "0-0": (0, 3),
        "1-0": (1, 4.2), "0-1": (1, 1.8),
        "2-0": (2, 5.2), "1-1": (2, 3), "0-2": (2, 0.8),
        "3-0": (3, 6.2), "2-1": (3, 4.2), "1-2": (3, 1.8), "0-3": (3, -0.2),
        "3-1": (4.2, 5.2), "1-3": (4.2, 0.8),
        "Deuce": (5.5, 3),
        "Ad-In": (6.8, 4.5), "Ad-Out": (6.8, 1.5),
        "Server Wins": (8.5, 5.5), "Returner Wins": (8.5, 0.5),
    }
    P = build_transition_matrix(p)
    fig = go.Figure()

    for i in range(N_TRANSIENT):
        for j in range(N_TOTAL):
            if P[i, j] > 0:
                x0, y0 = positions[ALL_STATES[i]]
                x1, y1 = positions[ALL_STATES[j]]
                if j >= N_TRANSIENT:
                    color = "#2e7d32" if ALL_STATES[j] == "Server Wins" else "#c62828"
                    width = 2.5
                else:
                    is_server = abs(P[i, j] - p) < 1e-10
                    color = "#43a047" if is_server else "#ef5350"
                    width = 1.5
                fig.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines', line=dict(width=width, color=color),
                    hoverinfo='skip', showlegend=False,
                ))

    for s in ALL_STATES:
        x, y = positions[s]
        if s == "Server Wins":       c, sz = "#2e7d32", 30
        elif s == "Returner Wins":   c, sz = "#c62828", 30
        elif s in DUEL_STATES:       c, sz = "#ef6c00", 22
        else:                        c, sz = "#1565c0", 17
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=sz, color=c, line=dict(width=2, color="white")),
            text=[s], textposition="top center",
            textfont=dict(size=10, color="#1a3a1a", family="IBM Plex Mono"),
            hovertext=f"<b>{s}</b><br>{TENNIS_SCORE_MAP.get(s,s)}",
            hoverinfo='text', showlegend=False,
        ))

    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Inter, sans-serif", color="#2a2a2a", size=12),
                      height=420, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def make_sankey(P, p):
    sources, targets, values, colors = [], [], [], []
    for i in range(N_TRANSIENT):
        for j in range(N_TOTAL):
            if P[i, j] > 0:
                sources.append(i); targets.append(j); values.append(P[i, j])
                if ALL_STATES[j] == "Server Wins":     colors.append("rgba(46,125,50,0.5)")
                elif ALL_STATES[j] == "Returner Wins": colors.append("rgba(198,40,40,0.4)")
                elif abs(P[i, j] - p) < 1e-10:         colors.append("rgba(46,125,50,0.25)")
                else:                                   colors.append("rgba(198,40,40,0.2)")
    nc = []
    for s in ALL_STATES:
        if s == "Server Wins":     nc.append("#2e7d32")
        elif s == "Returner Wins": nc.append("#c62828")
        elif s in DUEL_STATES:     nc.append("#ef6c00")
        else:                      nc.append("#1565c0")
    fig = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=18, line=dict(color="#ccc", width=0.5),
                  label=[f"{s}  ({TENNIS_SCORE_MAP[s]})" for s in ALL_STATES], color=nc),
        link=dict(source=sources, target=targets, value=values, color=colors),
    ))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#fafcfa",
                      font=dict(family="Inter, sans-serif", color="#2a2a2a", size=12),
                      height=480, margin=dict(l=10, r=10, t=20, b=10))
    return fig


def fmt_matrix(matrix, rows, cols, decimals=4):
    """Return a plain DataFrame with pre-formatted strings — avoids arrow_ serialization bug."""
    df = pd.DataFrame(matrix, columns=cols)
    df.insert(0, "State", rows)
    for c in cols:
        df[c] = df[c].map(lambda x: f"{x:.{decimals}f}")
    return df


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    p = st.slider("Server Point Win Probability (p)",
                   min_value=0.01, max_value=0.99, value=0.65, step=0.01,
                   help="Probability the server wins any individual point.")
    q_val = 1.0 - p
    st.markdown(f"""<div class="param-box">
        <span>p</span> <span>(server)</span> = <span class="val">{p:.2f}</span><br>
        <span>q</span> <span>(returner)</span> = <span class="val">{q_val:.2f}</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    n_trials = st.number_input("Monte Carlo Trials", min_value=100, max_value=1_000_000,
                                value=10_000, step=1000, help="Number of simulated games.")
    st.markdown("---")
    run_sim = st.button("▶  Run Simulation", use_container_width=True, type="primary")


# ═══════════════════════════════════════════════════════════════
# BUILD MODEL
# ═══════════════════════════════════════════════════════════════

P = build_transition_matrix(p)
Q, R, F, B, expected_steps = analytical_engine(P)


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("# 🎾 The Tennis Model")
st.markdown('<p class="subtitle">Absorbing Markov Chain — A Single Game of Tennis, Solved</p>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 1
# ═══════════════════════════════════════════════════════════════

st.markdown("## Phase 1 — State Space Definition")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("""<div class="phase-card">
        <h4>🏃 The Sprint</h4>
        <p>Linear race before both players reach 30. Points are absolute — no win-by-two required.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown(" ".join([f'<span class="state-badge">{s} · {TENNIS_SCORE_MAP[s]}</span>'
                          for s in SPRINT_STATES]), unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="phase-card">
        <h4>⚔️ The Duel</h4>
        <p>Win-by-two logic activates at 30-30. Only the point differential matters.</p>
    </div>""", unsafe_allow_html=True)
    st.markdown(" ".join([f'<span class="state-badge duel">{s} · {TENNIS_SCORE_MAP[s]}</span>'
                          for s in DUEL_STATES]), unsafe_allow_html=True)
    absorb_badges = " ".join([f'<span class="state-badge absorb">{s}</span>'
                               for s in ABSORBING_STATES])
    st.markdown(f"<div style='margin-top:8px'><b style='color:#2a2a2a;font-size:0.85rem'>"
                f"ABSORBING STATES</b><br>{absorb_badges}</div>", unsafe_allow_html=True)

st.markdown('<hr class="section-break">', unsafe_allow_html=True)

st.markdown("### State Transition Diagram")
st.markdown('<div class="info-callout">🟢 Green = server wins point (p) &nbsp;&nbsp; '
            '🔴 Red = returner wins point (q)</div>', unsafe_allow_html=True)
st.plotly_chart(make_state_diagram(p), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PHASE 2
# ═══════════════════════════════════════════════════════════════

st.markdown('<hr class="section-break">', unsafe_allow_html=True)
st.markdown("## Phase 2 — Transition Matrix (P)")
st.markdown('<div class="info-callout">P is partitioned into <b>Q</b> (transient → transient) '
            'and <b>R</b> (transient → absorbing) for canonical analysis.</div>',
            unsafe_allow_html=True)

with st.expander("Full Transition Matrix P  (17 x 17)", expanded=False):
    st.table(fmt_matrix(P, ALL_STATES, ALL_STATES))

c_q, c_r = st.columns(2, gap="large")
with c_q:
    with st.expander("Q Matrix  (Transient to Transient)", expanded=False):
        st.table(fmt_matrix(Q, TRANSIENT_STATES, TRANSIENT_STATES))
with c_r:
    with st.expander("R Matrix  (Transient to Absorbing)", expanded=False):
        st.table(fmt_matrix(R, TRANSIENT_STATES, ABSORBING_STATES))


# ═══════════════════════════════════════════════════════════════
# PHASE 3
# ═══════════════════════════════════════════════════════════════

st.markdown('<hr class="section-break">', unsafe_allow_html=True)
st.markdown("## Phase 3 — Analytical Engine")

st.markdown("### Key Results (starting from 0-0)")
m1, m2, m3 = st.columns(3)
with m1: st.metric("Server Win Probability", f"{B[0,0]:.6f}")
with m2: st.metric("Returner Win Probability", f"{B[0,1]:.6f}")
with m3: st.metric("Expected Points per Game", f"{expected_steps[0]:.4f}")

with st.expander("Fundamental Matrix  F = inv(I - Q)", expanded=False):
    st.markdown("F[i,j] = expected visits to state j, starting from state i.")
    st.table(fmt_matrix(F, TRANSIENT_STATES, TRANSIENT_STATES))

with st.expander("Absorption Probabilities  B = F x R", expanded=False):
    st.markdown("B[i,j] = probability of absorption into state j, starting from state i.")
    st.table(fmt_matrix(B, TRANSIENT_STATES, ABSORBING_STATES, decimals=6))

st.markdown("### Expected State Visits (from 0-0)")
visit_df = pd.DataFrame({
    "State": TRANSIENT_STATES,
    "Tennis Score": [TENNIS_SCORE_MAP[s] for s in TRANSIENT_STATES],
    "Expected Visits": F[0],
    "Phase": ["Sprint" if s in SPRINT_STATES else "Duel" for s in TRANSIENT_STATES],
})
fig_v = px.bar(visit_df, x="State", y="Expected Visits", color="Phase",
               color_discrete_map={"Sprint": "#1565c0", "Duel": "#ef6c00"},
               hover_data=["Tennis Score"])
fig_v.update_layout(**PLOT_LAYOUT, height=370,
                    xaxis=dict(title="State", **GRID), yaxis=dict(title="Expected Visits", **GRID),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)"))
st.plotly_chart(fig_v, use_container_width=True)

st.markdown("### Probability Flow (Sankey)")
st.plotly_chart(make_sankey(P, p), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# DUAL-ENGINE VALIDATION
# ═══════════════════════════════════════════════════════════════

st.markdown('<hr class="section-break">', unsafe_allow_html=True)
st.markdown("## Dual-Engine Validation — The Reconciliation")

if run_sim or "mc_results" in st.session_state:
    if run_sim:
        with st.spinner(f"Simulating {n_trials:,} games..."):
            mc = monte_carlo_engine(p, n_trials)
            st.session_state["mc_results"] = mc
            st.session_state["mc_p"] = p
            st.session_state["mc_n"] = n_trials
    else:
        mc = st.session_state["mc_results"]

    st.success(f"Simulation complete — **{st.session_state.get('mc_n', n_trials):,} trials** "
               f"at p = {st.session_state.get('mc_p', p):.2f}")

    # Win Prob
    st.markdown("### Win Probability Comparison")
    w1, w2 = st.columns(2, gap="large")
    with w1:
        st.markdown("**Server**")
        a1, b1 = st.columns(2)
        with a1: st.metric("Analytical", f"{B[0,0]:.6f}")
        with b1: st.metric("Monte Carlo", f"{mc['server_win_prob']:.6f}",
                            delta=f"{mc['server_win_prob']-B[0,0]:+.6f}")
    with w2:
        st.markdown("**Returner**")
        a2, b2 = st.columns(2)
        with a2: st.metric("Analytical", f"{B[0,1]:.6f}")
        with b2: st.metric("Monte Carlo", f"{mc['returner_win_prob']:.6f}",
                            delta=f"{mc['returner_win_prob']-B[0,1]:+.6f}")

    # Expected Points
    st.markdown("### Expected Points per Game")
    e1, e2, e3 = st.columns(3)
    with e1: st.metric("Analytical", f"{expected_steps[0]:.4f}")
    with e2: st.metric("Monte Carlo", f"{mc['expected_points']:.4f}")
    with e3: st.metric("Difference", f"{mc['expected_points']-expected_steps[0]:+.4f}")

    # State Visit Comparison
    st.markdown("### State Visit Comparison")
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="Analytical", x=TRANSIENT_STATES, y=F[0],
                             marker_color="#1565c0", opacity=0.85))
    fig_cmp.add_trace(go.Bar(name="Monte Carlo", x=TRANSIENT_STATES,
                             y=mc["state_visit_probs"], marker_color="#ef6c00", opacity=0.85))
    fig_cmp.update_layout(**PLOT_LAYOUT, barmode='group', height=380,
                          xaxis=dict(title="State", **GRID),
                          yaxis=dict(title="Expected Visits", **GRID),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_cmp, use_container_width=True)

    with st.expander("Detailed Comparison Table", expanded=False):
        cdf = pd.DataFrame({
            "State": TRANSIENT_STATES,
            "Score": [TENNIS_SCORE_MAP[s] for s in TRANSIENT_STATES],
            "Analytical": [f"{v:.4f}" for v in F[0]],
            "Monte Carlo": [f"{v:.4f}" for v in mc["state_visit_probs"]],
            "Difference": [f"{v:+.4f}" for v in (mc["state_visit_probs"] - F[0])],
        })
        st.table(cdf)

    # Game Length Distribution
    st.markdown("### Distribution of Game Length")
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=mc["point_counts"], nbinsx=30,
                                  marker_color="#43a047", opacity=0.8, name="Simulated"))
    fig_h.add_vline(x=expected_steps[0], line_dash="dash", line_color="#c62828", line_width=2,
                    annotation_text=f"Analytical: {expected_steps[0]:.2f}",
                    annotation_font=dict(color="#c62828", size=12))
    fig_h.add_vline(x=mc["expected_points"], line_dash="dot", line_color="#ef6c00", line_width=2,
                    annotation_text=f"MC: {mc['expected_points']:.2f}",
                    annotation_font=dict(color="#ef6c00", size=12))
    fig_h.update_layout(**PLOT_LAYOUT, height=370,
                        xaxis=dict(title="Points Played", **GRID),
                        yaxis=dict(title="Frequency", **GRID))
    st.plotly_chart(fig_h, use_container_width=True)

    # Convergence
    st.markdown("### Convergence — MC → Analytical")
    st.markdown('<div class="info-callout">As trials increase, the experimental running average '
                'converges onto the analytical prediction.</div>', unsafe_allow_html=True)

    pt = np.array(mc["point_counts"])
    run_pts = np.cumsum(pt) / np.arange(1, len(pt)+1)
    wn = np.array(mc["win_sequence"])
    run_win = np.cumsum(wn) / np.arange(1, len(wn)+1)
    si = np.unique(np.geomspace(1, len(run_pts), num=600).astype(int)) - 1

    tab1, tab2 = st.tabs(["Expected Points Convergence", "Win Probability Convergence"])
    with tab1:
        fc1 = go.Figure()
        fc1.add_trace(go.Scatter(x=si+1, y=run_pts[si], mode='lines', name='MC Running Avg',
                                  line=dict(color="#ef6c00", width=2)))
        fc1.add_hline(y=expected_steps[0], line_dash="dash", line_color="#2e7d32", line_width=2,
                      annotation_text=f"Analytical: {expected_steps[0]:.4f}",
                      annotation_font=dict(color="#2e7d32", size=12))
        fc1.update_layout(**PLOT_LAYOUT, height=370,
                          xaxis=dict(title="Trials (log)", type="log", **GRID),
                          yaxis=dict(title="Avg Points/Game", **GRID))
        st.plotly_chart(fc1, use_container_width=True)
    with tab2:
        fc2 = go.Figure()
        fc2.add_trace(go.Scatter(x=si+1, y=run_win[si], mode='lines', name='MC Running Win %',
                                  line=dict(color="#1565c0", width=2)))
        fc2.add_hline(y=B[0,0], line_dash="dash", line_color="#2e7d32", line_width=2,
                      annotation_text=f"Analytical: {B[0,0]:.6f}",
                      annotation_font=dict(color="#2e7d32", size=12))
        fc2.update_layout(**PLOT_LAYOUT, height=370,
                          xaxis=dict(title="Trials (log)", type="log", **GRID),
                          yaxis=dict(title="Server Win Probability", **GRID))
        st.plotly_chart(fc2, use_container_width=True)

else:
    st.markdown('<div class="info-callout">👈 Set parameters in the sidebar and press '
                '<b>Run Simulation</b> to launch the Monte Carlo engine.</div>',
                unsafe_allow_html=True)

# Footer
st.markdown('<hr class="section-break">', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;padding:1rem;">'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.78rem;color:#8a9a8a;">'
            'The Tennis Model · Absorbing Markov Chain Validator · Menlo School'
            '</span></div>', unsafe_allow_html=True)
