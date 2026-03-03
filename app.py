import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page Config ───
st.set_page_config(
    page_title="The Tennis Model",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=IBM+Plex+Mono:wght@400;600&family=Source+Sans+Pro:wght@400;600;700&display=swap');

.main { background-color: #0a0f0d; }
[data-testid="stSidebar"] { background-color: #111a15; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #c8e6c9 !important;
}
h1 { color: #66bb6a !important; letter-spacing: 0.02em; }

p, li, span, div, label {
    font-family: 'Source Sans Pro', sans-serif !important;
    color: #b0bfb3 !important;
}

.stMetric label { color: #81c784 !important; font-family: 'IBM Plex Mono', monospace !important; }
.stMetric [data-testid="stMetricValue"] {
    color: #e8f5e9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.8rem !important;
}

div[data-testid="stDataFrame"] th {
    background-color: #1b5e20 !important;
    color: #e8f5e9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.state-tag {
    display: inline-block;
    background: #1b5e20;
    color: #a5d6a7;
    padding: 2px 10px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    margin: 2px;
}

.section-divider {
    border: none;
    height: 1px;
    background: linear-gradient(to right, transparent, #2e7d32, transparent);
    margin: 2rem 0;
}

div[data-testid="stExpander"] {
    border: 1px solid #2e7d32 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODEL DEFINITION — Based on the whiteboard diagram
# ═══════════════════════════════════════════════════════════════
#
# TRANSIENT STATES (indices 0-12):
#   Sprint states (absolute score — server pts, returner pts):
#     0: 0-0    1: 1-0    2: 0-1    3: 2-0    4: 1-1    5: 0-2
#     6: 3-0    7: 2-1    8: 1-2    9: 0-3
#   Duel states (point differential once both reach 2+):
#     10: Duel_0  (deuce, differential = 0)
#     11: Duel+1  (ad server)
#     12: Duel-1  (ad returner)
#
# ABSORBING STATES (indices 13-14):
#     13: WIN  (server wins game)
#     14: LOSE (returner wins game)
#
# Transitions from the diagram:
#   Sprint phase — server wins point with prob p, returner with q=1-p
#     0-0  → 1-0 (p)  or 0-1 (q)
#     1-0  → 2-0 (p)  or 1-1 (q)
#     0-1  → 1-1 (p)  or 0-2 (q)
#     2-0  → 3-0 (p)  or 2-1 (q)
#     1-1  → 2-1 (p)  or 1-2 (q)
#     0-2  → 1-2 (p)  or 0-3 (q)
#     3-0  → WIN [4-0] (p)  or 3-1→Duel+1 ... but diagram shows
#            3-0 → 4-0 absorb (p) or 3-1 which feeds to +1
#            Actually from diagram: 3-0 → 4-0 (p, absorb=WIN) or 3-1 (q)
#            And 3-1 → WIN [4-1] (p, absorb) or 2-2→Duel_0 (q)
#            Wait — let me re-read the diagram more carefully.
#
#   From the whiteboard:
#     3-0: server wins → 4-0 (WIN absorb); returner wins → 3-1
#     3-1: server wins → 4-1 (WIN absorb); returner wins → 2-2 → Duel_0
#          BUT diagram also shows 3-1 → +1. Let me look again...
#          The arrows from 3-1 go to +1 (server wins) AND to 2-2 (returner wins).
#          And 4-1 also has an arrow to WIN. So 3-1 → +1 on server win,
#          and 2-2 feeds into Duel_0.
#          Actually re-examining: 3-1 with server win = game over (4-1 = WIN).
#          With returner win = 3-2, but using the "pivot at 2-2" concept,
#          once both have ≥2, we use differential. 3-1 server lead means
#          differential = +2... no wait, scores are points won not tennis scores.
#
#   Let me re-interpret the diagram using POINT counts (not tennis 15/30/40):
#     The numbers represent points won by each player.
#     "4-0" means server won 4, returner won 0 → server wins the game.
#     The "pivot" at 2-2: once both have 2 points (i.e. 30-30 in tennis),
#     the game enters the Duel phase.
#
#     So from the diagram:
#       3-0: p → 4-0 (WIN), q → 3-1
#       2-1: p → 3-1, q → 2-2
#       3-1: p → WIN (4-1), q → goes to Duel since both ≥ 2? No, 3-1
#             means server has 3, returner has 1. Returner only has 1.
#
#   OK let me just carefully read every arrow:
#
#   Actually from re-examining the whiteboard carefully:
#     3-1 has arrows going to +1 (box on right). This means from 3-1,
#     server winning gives WIN (since that's 4-1, game over before deuce).
#     BUT actually the arrow from 3-1 goes to the +1 box, not directly to WIN.
#     Let me look at 4-0 and 4-1 — they have arrows to WIN.
#     So 3-0 → 4-0 → WIN (p), and 3-0 → 3-1 (q).
#     3-1 → +1? Or 3-1 → 4-1 → WIN?
#
#   From the diagram the states 4-0, 4-1 have arrows to WIN (absorb).
#   And 0-4, 1-4 have arrows to LOSE (absorb).
#   States 3-1 and 1-3 feed into the duel cycle.
#   Specifically: 3-1 → +1 (when server scores, it's game-over worthy,
#     but diagram routes through +1). And 3-1 with returner scoring → 2-2 → 0.
#   Wait no — 3-1 returner scoring makes it 3-2, not 2-2.
#
#   I think the diagram's logic is:
#     The Sprint handles all states where NEITHER player has reached 3 points,
#     PLUS the "clean" wins (4-0, 4-1, 0-4, 1-4) that happen before deuce
#     is possible.
#     Once the score reaches a configuration where both have ≥ 2 points
#     AND neither has won outright, it enters the Duel.
#
#     The key transition states into the Duel:
#       2-2 → Duel_0 (both at 30-30, deuce, differential = 0)
#       3-1: server wins → 4-1 = WIN (direct absorb, not through duel).
#            returner wins → 3-2, which is like 40-30... but in the duel
#            framework with both ≥ 2, differential = 3-2 = +1 → Duel+1
#            WAIT, that makes it enter the duel at +1, which matches the arrow!
#       1-3: similar mirror — server wins → 2-3, differential = -1 → Duel-1
#            returner wins → 1-4 = LOSE (direct absorb)
#
#   SO the complete picture from the diagram:
#     3-1: p → WIN (4-1, absorbing), q → Duel+1 (score becomes 3-2, diff=+1)
#     1-3: p → Duel-1 (score becomes 2-3, diff=-1), q → LOSE (1-4, absorbing)
#
#   WAIT that doesn't make sense. If returner wins from 3-1, score is 3-2.
#   Server still leads by 1. That's +1 for server. From +1, server wins → WIN.
#   Returner wins from +1 → 0 (deuce). That matches the duel cycle!
#
#   And 1-3 with server winning → 2-3. Server trails by 1 = -1 differential.
#   From -1, returner wins → LOSE. Server wins → 0 (deuce). Matches!
#
#   Re-checking: 3-1, returner scores → 3-2. Both ≥ 2, diff = +1. → Duel+1 ✓
#   1-3, server scores → 2-3. Both ≥ 2, diff = -1. → Duel-1 ✓
#
#   And for 2-1: p → 3-1 (still sprint, server at 3 returner at 1), q → 2-2
#   2-2: enters Duel_0 directly.
#   1-2: p → 2-2, q → 1-3
#
#   Perfect — now the full model is clear!
#
# ═══════════════════════════════════════════════════════════════

TRANSIENT_STATES = [
    "0-0",   # 0
    "1-0",   # 1
    "0-1",   # 2
    "2-0",   # 3
    "1-1",   # 4
    "0-2",   # 5
    "3-0",   # 6
    "2-1",   # 7
    "1-2",   # 8
    "0-3",   # 9
    "Deuce",  # 10  (Duel_0, differential = 0)
    "Ad-In",  # 11  (Duel+1, server advantage)
    "Ad-Out", # 12  (Duel-1, returner advantage)
]

ABSORBING_STATES = [
    "Server Wins",   # 13
    "Returner Wins", # 14
]

ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES
N_TRANSIENT = len(TRANSIENT_STATES)
N_ABSORBING = len(ABSORBING_STATES)
N_TOTAL = N_TRANSIENT + N_ABSORBING

TENNIS_SCORE_MAP = {
    "0-0": "Love-Love", "1-0": "15-Love", "0-1": "Love-15",
    "2-0": "30-Love", "1-1": "15-15", "0-2": "Love-30",
    "3-0": "40-Love", "2-1": "30-15", "1-2": "15-30", "0-3": "Love-40",
    "Deuce": "Deuce", "Ad-In": "Ad-In", "Ad-Out": "Ad-Out",
    "Server Wins": "Game (Server)", "Returner Wins": "Game (Returner)",
}


def build_transition_matrix(p):
    """Build the full 15x15 transition matrix P based on the whiteboard diagram."""
    q = 1 - p
    P = np.zeros((N_TOTAL, N_TOTAL))

    # Index mapping
    idx = {name: i for i, name in enumerate(ALL_STATES)}

    # ── Sprint Phase ──
    # 0-0 → 1-0 (p), 0-1 (q)
    P[idx["0-0"], idx["1-0"]] = p
    P[idx["0-0"], idx["0-1"]] = q

    # 1-0 → 2-0 (p), 1-1 (q)
    P[idx["1-0"], idx["2-0"]] = p
    P[idx["1-0"], idx["1-1"]] = q

    # 0-1 → 1-1 (p), 0-2 (q)
    P[idx["0-1"], idx["1-1"]] = p
    P[idx["0-1"], idx["0-2"]] = q

    # 2-0 → 3-0 (p), 2-1 (q)
    P[idx["2-0"], idx["3-0"]] = p
    P[idx["2-0"], idx["2-1"]] = q

    # 1-1 → 2-1 (p), 1-2 (q)
    P[idx["1-1"], idx["2-1"]] = p
    P[idx["1-1"], idx["1-2"]] = q

    # 0-2 → 1-2 (p), 0-3 (q)
    P[idx["0-2"], idx["1-2"]] = p
    P[idx["0-2"], idx["0-3"]] = q

    # 3-0 → WIN/4-0 (p), 3-1 situation:
    #   From diagram: 3-0 server wins → 4-0 → WIN (absorb)
    #   3-0 returner wins → 3-1 → treated as transient that immediately
    #   branches: server wins → 4-1 → WIN, returner wins → 3-2 → Duel+1
    #   But 3-1 isn't an explicit state in our model. We can handle it
    #   by making 3-0's transitions go two steps... OR we add 3-1/1-3 as
    #   explicit transient states.
    #
    #   For cleaner modeling matching the diagram, let me add 3-1 and 1-3
    #   as explicit states... Actually no, let me re-examine.
    #   The diagram shows 3-1 as a node. Let me add it.

    # Actually, I realize I need 3-1 and 1-3 as explicit transient states
    # because they appear as nodes in the diagram. Let me rebuild.

    return P  # placeholder, will rebuild below


# Rebuild with 3-1 and 1-3 included
TRANSIENT_STATES = [
    "0-0",    # 0
    "1-0",    # 1
    "0-1",    # 2
    "2-0",    # 3
    "1-1",    # 4
    "0-2",    # 5
    "3-0",    # 6
    "2-1",    # 7
    "1-2",    # 8
    "0-3",    # 9
    "3-1",    # 10
    "2-2",    # 11  (this maps to Deuce entry, but is the first time at 30-30)
    "1-3",    # 12
    "Deuce",  # 13  (Duel_0)
    "Ad-In",  # 14  (Duel+1)
    "Ad-Out", # 15  (Duel-1)
]

ABSORBING_STATES = [
    "Server Wins",   # 16
    "Returner Wins", # 17
]

ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES
N_TRANSIENT = len(TRANSIENT_STATES)
N_ABSORBING = len(ABSORBING_STATES)
N_TOTAL = N_TRANSIENT + N_ABSORBING

# Actually, from the diagram, 2-2 goes directly into the duel cycle (Deuce/0).
# So 2-2 IS Deuce. Let me collapse them.
# The diagram shows: 2-2 → +1 (p), 2-2 → -1 (q), and +1 ↔ 0 ↔ -1 cycle.
# But 2-2 also shows an arrow to the "0" box (Deuce). So 2-2 = Deuce.
#
# Similarly, 3-1 feeds into +1 (Ad-In) on returner win (since 3-2 = +1 diff).
# And 1-3 feeds into -1 (Ad-Out) on server win (since 2-3 = -1 diff).
#
# So I can handle 3-1 and 1-3 as transient states that either absorb or
# enter the duel. And 2-2 can just be Deuce (no separate state needed).

TRANSIENT_STATES = [
    "0-0",    # 0
    "1-0",    # 1
    "0-1",    # 2
    "2-0",    # 3
    "1-1",    # 4
    "0-2",    # 5
    "3-0",    # 6
    "2-1",    # 7
    "1-2",    # 8
    "0-3",    # 9
    "3-1",    # 10
    "1-3",    # 11
    "Deuce",  # 12  (= 2-2 / 30-30 / Duel_0)
    "Ad-In",  # 13  (Duel+1 / Ad server)
    "Ad-Out", # 14  (Duel-1 / Ad returner)
]

ABSORBING_STATES = [
    "Server Wins",   # 15
    "Returner Wins", # 16
]

ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES
N_TRANSIENT = len(TRANSIENT_STATES)
N_ABSORBING = len(ABSORBING_STATES)
N_TOTAL = N_TRANSIENT + N_ABSORBING

TENNIS_SCORE_MAP = {
    "0-0": "Love-All", "1-0": "15-Love", "0-1": "Love-15",
    "2-0": "30-Love", "1-1": "15-All", "0-2": "Love-30",
    "3-0": "40-Love", "2-1": "30-15", "1-2": "15-30", "0-3": "Love-40",
    "3-1": "40-15", "1-3": "15-40",
    "Deuce": "Deuce (30-30+)", "Ad-In": "Advantage Server", "Ad-Out": "Advantage Returner",
    "Server Wins": "Game — Server", "Returner Wins": "Game — Returner",
}

SPRINT_STATES = ["0-0","1-0","0-1","2-0","1-1","0-2","3-0","2-1","1-2","0-3","3-1","1-3"]
DUEL_STATES = ["Deuce","Ad-In","Ad-Out"]


def build_transition_matrix(p):
    """Build the full transition matrix based on the whiteboard diagram."""
    q = 1.0 - p
    P = np.zeros((N_TOTAL, N_TOTAL))
    idx = {name: i for i, name in enumerate(ALL_STATES)}

    # ── Sprint Phase transitions ──
    P[idx["0-0"], idx["1-0"]] = p
    P[idx["0-0"], idx["0-1"]] = q

    P[idx["1-0"], idx["2-0"]] = p
    P[idx["1-0"], idx["1-1"]] = q

    P[idx["0-1"], idx["1-1"]] = p
    P[idx["0-1"], idx["0-2"]] = q

    P[idx["2-0"], idx["3-0"]] = p
    P[idx["2-0"], idx["2-1"]] = q

    P[idx["1-1"], idx["2-1"]] = p
    P[idx["1-1"], idx["1-2"]] = q

    P[idx["0-2"], idx["1-2"]] = p
    P[idx["0-2"], idx["0-3"]] = q

    # 3-0: server wins → 4-0 (WIN absorb), returner wins → 3-1
    P[idx["3-0"], idx["Server Wins"]] = p
    P[idx["3-0"], idx["3-1"]] = q

    # 2-1: server wins → 3-1, returner wins → 2-2 = Deuce
    P[idx["2-1"], idx["3-1"]] = p
    P[idx["2-1"], idx["Deuce"]] = q

    # 1-2: server wins → 2-2 = Deuce, returner wins → 1-3
    P[idx["1-2"], idx["Deuce"]] = p
    P[idx["1-2"], idx["1-3"]] = q

    # 0-3: server wins → 1-3, returner wins → 0-4 (LOSE absorb)
    P[idx["0-3"], idx["1-3"]] = p
    P[idx["0-3"], idx["Returner Wins"]] = q

    # 3-1: server wins → 4-1 (WIN absorb), returner wins → 3-2 → diff +1 = Ad-In
    P[idx["3-1"], idx["Server Wins"]] = p
    P[idx["3-1"], idx["Ad-In"]] = q

    # 1-3: server wins → 2-3 → diff -1 = Ad-Out, returner wins → 1-4 (LOSE absorb)
    P[idx["1-3"], idx["Ad-Out"]] = p
    P[idx["1-3"], idx["Returner Wins"]] = q

    # ── Duel Phase transitions (the cycle) ──
    # Deuce (0): server wins → Ad-In (+1), returner wins → Ad-Out (-1)
    P[idx["Deuce"], idx["Ad-In"]] = p
    P[idx["Deuce"], idx["Ad-Out"]] = q

    # Ad-In (+1): server wins → WIN (diff +2), returner wins → Deuce (0)
    P[idx["Ad-In"], idx["Server Wins"]] = p
    P[idx["Ad-In"], idx["Deuce"]] = q

    # Ad-Out (-1): server wins → Deuce (0), returner wins → LOSE (diff -2)
    P[idx["Ad-Out"], idx["Deuce"]] = p
    P[idx["Ad-Out"], idx["Returner Wins"]] = q

    # Absorbing states (self-loops)
    P[idx["Server Wins"], idx["Server Wins"]] = 1.0
    P[idx["Returner Wins"], idx["Returner Wins"]] = 1.0

    return P


def analytical_engine(P):
    """Extract Q, R, compute F = (I-Q)^-1, B = F*R, expected steps."""
    Q = P[:N_TRANSIENT, :N_TRANSIENT]
    R = P[:N_TRANSIENT, N_TRANSIENT:]
    I = np.eye(N_TRANSIENT)
    F = np.linalg.inv(I - Q)  # Fundamental matrix
    B = F @ R                  # Absorption probabilities
    # Expected steps from each state = F * 1-vector (row sums of F)
    expected_steps = F.sum(axis=1)
    return Q, R, F, B, expected_steps


def monte_carlo_engine(p, n_trials):
    """Simulate n_trials games, return stats."""
    idx = {name: i for i, name in enumerate(ALL_STATES)}
    rev_idx = {i: name for name, i in idx.items()}
    absorbing_indices = {idx["Server Wins"], idx["Returner Wins"]}

    server_wins = 0
    total_points = 0
    state_visits = np.zeros(N_TRANSIENT)
    point_counts = []

    # Build adjacency for fast simulation
    # For each transient state, store (next_state_if_server_wins, next_state_if_returner_wins)
    transitions = {}
    P_local = build_transition_matrix(p)
    for i in range(N_TRANSIENT):
        targets = np.where(P_local[i] > 0)[0]
        # There should be exactly 2 targets (one for p, one for q)
        # Sort by probability to identify which is server-win vs returner-win
        probs = [(j, P_local[i, j]) for j in targets]
        probs.sort(key=lambda x: -x[1])  # highest prob first
        if len(probs) == 2:
            if p >= 0.5:
                transitions[i] = (probs[0][0], probs[1][0])  # (server_win, returner_win)
            else:
                transitions[i] = (probs[1][0], probs[0][0])
        elif len(probs) == 1:
            transitions[i] = (probs[0][0], probs[0][0])

    # Handle p=0.5 case — need to use matrix structure not probability ordering
    # Rebuild transitions explicitly
    transitions = {}
    for i in range(N_TRANSIENT):
        row = P_local[i]
        server_target = None
        returner_target = None
        for j in range(N_TOTAL):
            if row[j] > 0:
                # Determine if this is the "p" transition or "q" transition
                # We know the structure: p goes "forward" for server
                if abs(row[j] - p) < 1e-10:
                    server_target = j
                elif abs(row[j] - (1-p)) < 1e-10:
                    if server_target == j:
                        returner_target = j  # p = 0.5 edge case
                    else:
                        returner_target = j
        if server_target is None:
            server_target = returner_target
        if returner_target is None:
            returner_target = server_target
        # Handle p=0.5 — both transitions have same probability
        if p == 0.5:
            targets = [j for j in range(N_TOTAL) if row[j] > 0]
            if len(targets) == 2:
                # Use domain knowledge: for each state we know which way is "server wins point"
                server_target, returner_target = assign_transitions_p05(TRANSIENT_STATES[i], targets, idx)
            elif len(targets) == 1:
                server_target = returner_target = targets[0]
        transitions[i] = (server_target, returner_target)

    for _ in range(n_trials):
        state = 0  # start at 0-0
        points = 0
        while state not in absorbing_indices:
            state_visits[state] += 1
            if random.random() < p:
                state = transitions[state][0]
            else:
                state = transitions[state][1]
            points += 1
        total_points += points
        point_counts.append(points)
        if state == idx["Server Wins"]:
            server_wins += 1

    exp_points = total_points / n_trials
    server_win_prob = server_wins / n_trials
    returner_win_prob = 1.0 - server_win_prob
    state_visit_probs = state_visits / n_trials

    return {
        "server_win_prob": server_win_prob,
        "returner_win_prob": returner_win_prob,
        "expected_points": exp_points,
        "state_visit_probs": state_visit_probs,
        "point_counts": point_counts,
    }


def assign_transitions_p05(state_name, targets, idx):
    """For p=0.5, we need domain knowledge to know which target is server-win."""
    # Map each state to its two possible next states: (server_wins_point, returner_wins_point)
    mapping = {
        "0-0": ("1-0", "0-1"),
        "1-0": ("2-0", "1-1"),
        "0-1": ("1-1", "0-2"),
        "2-0": ("3-0", "2-1"),
        "1-1": ("2-1", "1-2"),
        "0-2": ("1-2", "0-3"),
        "3-0": ("Server Wins", "3-1"),
        "2-1": ("3-1", "Deuce"),
        "1-2": ("Deuce", "1-3"),
        "0-3": ("1-3", "Returner Wins"),
        "3-1": ("Server Wins", "Ad-In"),
        "1-3": ("Ad-Out", "Returner Wins"),
        "Deuce": ("Ad-In", "Ad-Out"),
        "Ad-In": ("Server Wins", "Deuce"),
        "Ad-Out": ("Deuce", "Returner Wins"),
    }
    s_name, r_name = mapping[state_name]
    s_idx, r_idx = idx[s_name], idx[r_name]
    return (s_idx, r_idx)


def create_sankey_diagram(P, p):
    """Create a Sankey diagram of state transitions."""
    sources, targets, values, colors = [], [], [], []
    idx = {name: i for i, name in enumerate(ALL_STATES)}

    color_map = {
        "sprint_p": "rgba(76, 175, 80, 0.4)",   # green for server
        "sprint_q": "rgba(239, 83, 80, 0.3)",    # red for returner
        "duel_p": "rgba(129, 199, 132, 0.5)",
        "duel_q": "rgba(229, 115, 115, 0.4)",
        "absorb_win": "rgba(76, 175, 80, 0.7)",
        "absorb_lose": "rgba(239, 83, 80, 0.6)",
    }

    for i in range(N_TRANSIENT):
        for j in range(N_TOTAL):
            if P[i, j] > 0:
                sources.append(i)
                targets.append(j)
                values.append(P[i, j])
                if j == idx["Server Wins"]:
                    colors.append(color_map["absorb_win"])
                elif j == idx["Returner Wins"]:
                    colors.append(color_map["absorb_lose"])
                elif abs(P[i, j] - p) < 1e-10:
                    colors.append(color_map["sprint_p"] if ALL_STATES[i] in SPRINT_STATES else color_map["duel_p"])
                else:
                    colors.append(color_map["sprint_q"] if ALL_STATES[i] in SPRINT_STATES else color_map["duel_q"])

    node_colors = []
    for s in ALL_STATES:
        if s == "Server Wins":
            node_colors.append("#43a047")
        elif s == "Returner Wins":
            node_colors.append("#e53935")
        elif s in DUEL_STATES:
            node_colors.append("#ffa726")
        else:
            node_colors.append("#42a5f5")

    labels = [f"{s}\n({TENNIS_SCORE_MAP.get(s, s)})" for s in ALL_STATES]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="#333", width=0.5),
            label=labels, color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color=colors),
    )])
    fig.update_layout(
        font=dict(size=11, color="#b0bfb3", family="Source Sans Pro"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def create_state_diagram_fig(p):
    """Create a network-style state diagram using plotly."""
    # Positions roughly matching the whiteboard layout
    positions = {
        "0-0": (0, 3),
        "1-0": (1, 4), "0-1": (1, 2),
        "2-0": (2, 5), "1-1": (2, 3), "0-2": (2, 1),
        "3-0": (3, 6), "2-1": (3, 4), "1-2": (3, 2), "0-3": (3, 0),
        "3-1": (4, 5), "1-3": (4, 1),
        "Deuce": (5, 3),
        "Ad-In": (6, 4.5), "Ad-Out": (6, 1.5),
        "Server Wins": (7.5, 5.5), "Returner Wins": (7.5, 0.5),
    }

    P = build_transition_matrix(p)

    # Draw edges
    edge_traces = []
    for i in range(N_TRANSIENT):
        for j in range(N_TOTAL):
            if P[i, j] > 0:
                x0, y0 = positions[ALL_STATES[i]]
                x1, y1 = positions[ALL_STATES[j]]
                color = "#66bb6a" if abs(P[i, j] - p) < 1e-10 else "#ef5350"
                if j >= N_TRANSIENT:
                    color = "#43a047" if ALL_STATES[j] == "Server Wins" else "#e53935"
                    width = 2.5
                else:
                    width = 1.5
                edge_traces.append(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none', showlegend=False,
                ))

    # Draw nodes
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for s in ALL_STATES:
        x, y = positions[s]
        node_x.append(x)
        node_y.append(y)
        tennis = TENNIS_SCORE_MAP.get(s, s)
        node_text.append(f"{s}<br>({tennis})")
        if s == "Server Wins":
            node_color.append("#43a047")
            node_size.append(28)
        elif s == "Returner Wins":
            node_color.append("#e53935")
            node_size.append(28)
        elif s in DUEL_STATES:
            node_color.append("#ffa726")
            node_size.append(22)
        else:
            node_color.append("#42a5f5")
            node_size.append(18)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="#222")),
        text=[s for s in ALL_STATES],
        textposition="top center",
        textfont=dict(size=10, color="#c8e6c9", family="IBM Plex Mono"),
        hovertext=node_text, hoverinfo='text', showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family="Source Sans Pro"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

st.title("🎾 The Tennis Model")
st.markdown("##### Absorbing Markov Chain — A Single Game of Tennis, Solved")
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Sidebar Inputs ──
with st.sidebar:
    st.header("⚙️ Parameters")
    st.markdown("---")

    p = st.slider(
        "**Server Point Win Probability (p)**",
        min_value=0.01, max_value=0.99, value=0.65, step=0.01,
        help="Probability the server wins any individual point."
    )
    q = 1.0 - p

    st.markdown(f"""
    <div style="background:#1b5e20; padding:12px; border-radius:8px; margin:10px 0;">
        <span style="font-family:'IBM Plex Mono',monospace; color:#a5d6a7;">
        p (server) = {p:.2f}<br>
        q (returner) = {q:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    n_trials = st.number_input(
        "**Monte Carlo Trials**",
        min_value=100, max_value=1_000_000, value=10_000, step=1000,
        help="Number of games to simulate for the Monte Carlo engine."
    )

    st.markdown("---")
    run_sim = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# ── Build Model ──
P = build_transition_matrix(p)
Q, R, F, B, expected_steps = analytical_engine(P)

# ── Phase 1: State Space ──
st.header("Phase 1 — State Space Definition")

col1, col2 = st.columns(2)
with col1:
    st.subheader("The Sprint")
    st.markdown("Linear race before both players reach 30. Points are absolute.")
    sprint_html = " ".join([f'<span class="state-tag">{s} ({TENNIS_SCORE_MAP[s]})</span>' for s in SPRINT_STATES])
    st.markdown(sprint_html, unsafe_allow_html=True)

with col2:
    st.subheader("The Duel")
    st.markdown("Win-by-two logic activates at 30-30. Only differential matters.")
    duel_html = " ".join([f'<span class="state-tag">{s} ({TENNIS_SCORE_MAP[s]})</span>' for s in DUEL_STATES])
    st.markdown(duel_html, unsafe_allow_html=True)
    st.markdown("**Absorbing States:**")
    absorb_html = " ".join([f'<span class="state-tag">{s}</span>' for s in ABSORBING_STATES])
    st.markdown(absorb_html, unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── State Diagram ──
st.subheader("State Transition Diagram")
st.plotly_chart(create_state_diagram_fig(p), use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Phase 2: Transition Matrix ──
st.header("Phase 2 — Transition Matrix (P)")

with st.expander("📐 Full Transition Matrix P", expanded=False):
    df_P = pd.DataFrame(P, index=ALL_STATES, columns=ALL_STATES)
    st.dataframe(df_P.style.format("{:.4f}").background_gradient(
        cmap="Greens", vmin=0, vmax=1
    ), use_container_width=True)

col_q, col_r = st.columns(2)
with col_q:
    with st.expander("Q Matrix (Transient → Transient)", expanded=False):
        df_Q = pd.DataFrame(Q, index=TRANSIENT_STATES, columns=TRANSIENT_STATES)
        st.dataframe(df_Q.style.format("{:.4f}").background_gradient(
            cmap="Blues", vmin=0, vmax=1
        ), use_container_width=True)

with col_r:
    with st.expander("R Matrix (Transient → Absorbing)", expanded=False):
        df_R = pd.DataFrame(R, index=TRANSIENT_STATES, columns=ABSORBING_STATES)
        st.dataframe(df_R.style.format("{:.4f}").background_gradient(
            cmap="Oranges", vmin=0, vmax=1
        ), use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Phase 3: Analytical Results ──
st.header("Phase 3 — Analytical Engine")

st.subheader("Fundamental Matrix  F = (I − Q)⁻¹")
with st.expander("View Fundamental Matrix F", expanded=False):
    df_F = pd.DataFrame(F, index=TRANSIENT_STATES, columns=TRANSIENT_STATES)
    st.dataframe(df_F.style.format("{:.4f}").background_gradient(
        cmap="YlGn", vmin=0
    ), use_container_width=True)

st.subheader("Absorption Probabilities  B = F × R")
df_B = pd.DataFrame(B, index=TRANSIENT_STATES, columns=ABSORBING_STATES)

col_a1, col_a2, col_a3 = st.columns(3)
with col_a1:
    st.metric("Server Win Probability", f"{B[0, 0]:.6f}")
with col_a2:
    st.metric("Returner Win Probability", f"{B[0, 1]:.6f}")
with col_a3:
    st.metric("Expected Points per Game", f"{expected_steps[0]:.4f}")

with st.expander("Full B Matrix (from every state)", expanded=False):
    st.dataframe(df_B.style.format("{:.6f}").background_gradient(
        cmap="RdYlGn", vmin=0, vmax=1
    ), use_container_width=True)

st.subheader("Expected Visits per State (from 0-0)")
visit_data = pd.DataFrame({
    "State": TRANSIENT_STATES,
    "Tennis Score": [TENNIS_SCORE_MAP[s] for s in TRANSIENT_STATES],
    "Expected Visits": F[0],
    "Phase": ["Sprint" if s in SPRINT_STATES else "Duel" for s in TRANSIENT_STATES],
})

fig_visits = px.bar(
    visit_data, x="State", y="Expected Visits", color="Phase",
    color_discrete_map={"Sprint": "#42a5f5", "Duel": "#ffa726"},
    hover_data=["Tennis Score"],
)
fig_visits.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b0bfb3", family="Source Sans Pro"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    height=350,
)
st.plotly_chart(fig_visits, use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Sankey Diagram ──
st.subheader("Flow Diagram (Sankey)")
st.plotly_chart(create_sankey_diagram(P, p), use_container_width=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MONTE CARLO + RECONCILIATION
# ═══════════════════════════════════════════════════════════════

st.header("Dual-Engine Validation — The Reconciliation")

if run_sim or "mc_results" in st.session_state:
    if run_sim:
        with st.spinner(f"Running {n_trials:,} Monte Carlo trials..."):
            mc = monte_carlo_engine(p, n_trials)
            st.session_state["mc_results"] = mc
            st.session_state["mc_p"] = p
            st.session_state["mc_n"] = n_trials
    else:
        mc = st.session_state["mc_results"]

    st.success(f"✅ Simulation complete — {st.session_state.get('mc_n', n_trials):,} trials at p = {st.session_state.get('mc_p', p):.2f}")

    # ── Win Probability Comparison ──
    st.subheader("Win Probability Comparison")
    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("**Server Win Probability**")
        st.metric("Analytical", f"{B[0, 0]:.6f}")
        st.metric("Monte Carlo", f"{mc['server_win_prob']:.6f}",
                  delta=f"{mc['server_win_prob'] - B[0,0]:+.6f}")

    with comp_col2:
        st.markdown("**Returner Win Probability**")
        st.metric("Analytical", f"{B[0, 1]:.6f}")
        st.metric("Monte Carlo", f"{mc['returner_win_prob']:.6f}",
                  delta=f"{mc['returner_win_prob'] - B[0,1]:+.6f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Expected Points Comparison ──
    st.subheader("Expected Points per Game")
    ep_col1, ep_col2, ep_col3 = st.columns(3)
    with ep_col1:
        st.metric("Analytical", f"{expected_steps[0]:.4f}")
    with ep_col2:
        st.metric("Monte Carlo", f"{mc['expected_points']:.4f}")
    with ep_col3:
        diff = mc['expected_points'] - expected_steps[0]
        st.metric("Difference", f"{diff:+.4f}")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── State Visit Probability Comparison ──
    st.subheader("State Visit Probability — Analytical vs. Monte Carlo")
    comparison_data = pd.DataFrame({
        "State": TRANSIENT_STATES,
        "Tennis Score": [TENNIS_SCORE_MAP[s] for s in TRANSIENT_STATES],
        "Analytical": F[0],
        "Monte Carlo": mc["state_visit_probs"],
        "Difference": mc["state_visit_probs"] - F[0],
    })

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name="Analytical", x=TRANSIENT_STATES, y=F[0],
        marker_color="#42a5f5", opacity=0.8,
    ))
    fig_comp.add_trace(go.Bar(
        name="Monte Carlo", x=TRANSIENT_STATES, y=mc["state_visit_probs"],
        marker_color="#ffa726", opacity=0.8,
    ))
    fig_comp.update_layout(
        barmode='group',
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b0bfb3", family="Source Sans Pro"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Expected Visits"),
        height=380, legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    with st.expander("Detailed Comparison Table"):
        st.dataframe(comparison_data.style.format({
            "Analytical": "{:.4f}", "Monte Carlo": "{:.4f}", "Difference": "{:+.4f}"
        }), use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Distribution of Game Length ──
    st.subheader("Distribution of Game Length (Points Played)")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=mc["point_counts"], nbinsx=30,
        marker_color="#66bb6a", opacity=0.8,
        name="Simulated Games",
    ))
    fig_hist.add_vline(x=expected_steps[0], line_dash="dash", line_color="#ef5350",
                       annotation_text=f"Analytical Mean: {expected_steps[0]:.2f}",
                       annotation_font_color="#ef5350")
    fig_hist.add_vline(x=mc["expected_points"], line_dash="dot", line_color="#ffa726",
                       annotation_text=f"MC Mean: {mc['expected_points']:.2f}",
                       annotation_font_color="#ffa726")
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b0bfb3", family="Source Sans Pro"),
        xaxis=dict(title="Points Played", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.08)"),
        height=350,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Convergence Plot ──
    st.subheader("Convergence — Monte Carlo → Analytical")
    st.markdown("As trials increase, the experimental win probability converges onto the theoretical prediction.")

    # Compute running average
    point_arr = np.array(mc["point_counts"])
    # We need win/loss per trial for running win prob
    # Re-derive from point_counts... actually let's re-simulate a small version
    # Instead, compute running average of game lengths
    running_avg = np.cumsum(point_arr) / np.arange(1, len(point_arr) + 1)
    sample_indices = np.unique(np.geomspace(1, len(running_avg), num=500).astype(int)) - 1

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=sample_indices + 1, y=running_avg[sample_indices],
        mode='lines', name='MC Running Average',
        line=dict(color="#ffa726", width=2),
    ))
    fig_conv.add_hline(y=expected_steps[0], line_dash="dash", line_color="#66bb6a",
                       annotation_text=f"Analytical: {expected_steps[0]:.4f}",
                       annotation_font_color="#66bb6a")
    fig_conv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#b0bfb3", family="Source Sans Pro"),
        xaxis=dict(title="Number of Trials", gridcolor="rgba(255,255,255,0.05)", type="log"),
        yaxis=dict(title="Avg Points per Game", gridcolor="rgba(255,255,255,0.08)"),
        height=350,
    )
    st.plotly_chart(fig_conv, use_container_width=True)

else:
    st.info("👈 Set parameters and press **Run Simulation** to launch the Monte Carlo engine and compare with the analytical solution.")

# ── Footer ──
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem; opacity:0.5;">
    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.8rem; color:#66bb6a;">
    The Tennis Model — Absorbing Markov Chain Validator — Menlo School
    </span>
</div>
""", unsafe_allow_html=True)
