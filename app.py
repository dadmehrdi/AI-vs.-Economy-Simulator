import streamlit as st
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# Page + Styling
# -----------------------------
st.set_page_config(page_title="AI Economy Simulator", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.1rem;}
      .big-title {font-size: 2.2rem; font-weight: 850; letter-spacing: -0.02em;}
      .subtitle {opacity: 0.85; margin-top: -0.35rem;}
      .pill {display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
             background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.28);
             margin-right: 0.4rem; font-size: 0.85rem;}
      .panel {padding: 1rem; border-radius: 18px; background: rgba(255,255,255,0.03);
              border: 1px solid rgba(255,255,255,0.08);}
      .small {opacity: 0.75; font-size: 0.92rem;}
      code {font-size: 0.92em;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="big-title">🌍🤖 AI Economy Simulator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A toy macro sandbox: productivity distributions, AI adoption, displacement vs. reskilling, and GDP/employment over time.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<span class="pill">Population</span>'
    '<span class="pill">Firms (stylized)</span>'
    '<span class="pill">Savings → Investment</span>'
    '<span class="pill">AI Adoption</span>'
    '<span class="pill">Reskilling</span>'
    '<span class="pill">GDP + Employment</span>',
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
def _nice_line(x, ys, names, title, ylab):
    fig = go.Figure()
    for y, nm in zip(ys, names):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=nm, line=dict(width=3)))
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        template="plotly_dark",
        xaxis_title="Year",
        yaxis_title=ylab,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

def beta_from_mean_conc(mean, conc):
    """
    Convert mean in (0,1) and concentration > 2 into alpha,beta for a Beta distribution.
    conc ~ alpha+beta (bigger = tighter around mean)
    """
    mean = np.clip(mean, 1e-4, 1 - 1e-4)
    conc = max(conc, 2.1)
    a = mean * conc
    b = (1 - mean) * conc
    return a, b

def adoption_curve(prev, rate):
    """Bounded adoption: A_t = 1 - (1 - A_{t-1})*(1 - rate)"""
    return 1.0 - (1.0 - prev) * (1.0 - rate)

@st.cache_data(show_spinner=False)
def simulate_economy(
    seed: int,
    years: int,
    pop_millions: float,
    firms_ratio: float,
    # worker distributions
    prod_sigma: float,
    auto_mean: float,
    auto_conc: float,
    init_unemp: float,
    # AI dynamics
    ai_adoption_rate: float,
    replace_share: float,      # fraction of automated tasks that replace humans
    augment_strength: float,   # boost to still-employed humans as AI adoption rises
    ai_prod_mult: float,       # productivity multiplier of AI on automated tasks
    # reskilling
    reskill_rate: float,
    reskill_effect: float,     # makes workers less automatable after reskill
    upskill_gain: float,       # raises productivity after reskill
    # macro plumbing
    savings_rate: float,
    dep_rate: float,
    capital_boost: float,
    firm_creation_rate: float,
    firm_capital_cost: float,
    sample_workers: int,
):
    rng = np.random.default_rng(seed)

    # Scale: we simulate sample_workers agents, and scale totals up to population.
    L = pop_millions * 1_000_000.0
    n = int(sample_workers)
    scale = L / n

    # Firms (stylized count, not explicit firm agents)
    firm0 = max(1.0, L * firms_ratio)
    firms = firm0

    # Workers: productivity ~ lognormal, normalized to mean 1
    prod = rng.lognormal(mean=0.0, sigma=prod_sigma, size=n)
    prod = prod / (prod.mean() + 1e-12)

    # Automatability score ~ Beta (0..1). Higher = easier to automate.
    aa, bb = beta_from_mean_conc(auto_mean, auto_conc)
    auto = rng.beta(aa, bb, size=n)

    # Employment state
    employed = rng.random(n) > init_unemp

    # State vars over time
    A = 0.0            # AI adoption / automated-task share
    K = 1.0            # capital index
    Y0 = None          # store initial GDP for scaling capital accumulation

    yrs = np.arange(years + 1)
    gdp = np.zeros(years + 1)
    emp_rate = np.zeros(years + 1)
    cons = np.zeros(years + 1)
    inv = np.zeros(years + 1)
    ai_share = np.zeros(years + 1)

    def tfp_from_capital(K_):
        # Simple saturating capital productivity: 1 + c * log(1+K)
        return 1.0 + capital_boost * np.log1p(K_)

    for t in range(years + 1):
        # Determine which tasks are "automated" at this adoption level A:
        # automate the highest-automatability tasks first
        if A <= 0:
            automated_mask = np.zeros(n, dtype=bool)
        else:
            cutoff = np.quantile(auto, 1.0 - A)
            automated_mask = auto >= cutoff

        # Human output: sum productivity of employed humans, boosted by augmentation
        aug = 1.0 + augment_strength * A * (1.0 - replace_share)
        human_output = prod[employed].sum() * aug

        # AI output: automated tasks produce output even if displaced humans are unemployed.
        # Approx: use mean productivity of automated-task group times count, with multiplier.
        if automated_mask.any():
            auto_prod_mean = prod[automated_mask].mean()
            ai_output = automated_mask.sum() * auto_prod_mean * ai_prod_mult
        else:
            ai_output = 0.0

        # Total output (GDP) with capital TFP
        Y = (human_output + ai_output) * tfp_from_capital(K) * scale
        gdp[t] = Y

        # set baseline GDP once (for capital scaling)
        if Y0 is None:
            Y0 = Y + 1e-12

        # Employment
        emp_rate[t] = employed.mean()

        # AI share of output (rough)
        ai_share[t] = ai_output / (human_output + ai_output + 1e-12)

        # -----------------------------
        # Macro identity (closed economy, no gov, no trade):
        # Y = C + I
        # Choose savings rate s, then:
        # C = (1-s)Y
        # I = sY  (and by definition S = Y - C = I)
        # -----------------------------
        C = (1.0 - savings_rate) * Y
        I = savings_rate * Y
        cons[t] = C
        inv[t] = I

        # --- Transition to next period ---
        if t == years:
            break

        # 1) AI adoption update
        A_next = adoption_curve(A, ai_adoption_rate)

        # 2) Displacement using incremental adoption
        if A_next <= 0:
            automated_next = np.zeros(n, dtype=bool)
        else:
            cutoff_next = np.quantile(auto, 1.0 - A_next)
            automated_next = auto >= cutoff_next

        newly_automated = automated_next & (~automated_mask)
        to_consider = newly_automated & employed
        if to_consider.any():
            displace_draw = rng.random(n) < replace_share
            displaced = to_consider & displace_draw
            employed[displaced] = False

        # 3) Reskilling / re-entry
        unemployed_idx = np.where(~employed)[0]
        n_rehire = int(reskill_rate * len(unemployed_idx))
        if n_rehire > 0:
            rehired = rng.choice(unemployed_idx, size=n_rehire, replace=False)
            employed[rehired] = True
            auto[rehired] = np.clip(auto[rehired] * (1.0 - reskill_effect), 0.0, 1.0)
            prod[rehired] = prod[rehired] * (1.0 + upskill_gain)

        # 4) Capital accumulation (simple + consistent scaling)
        # K_{t+1} = (1-δ)K_t + I_t / Y0
        K = (1.0 - dep_rate) * K + (I / Y0)

        # (Optional stylized) firm creation proportional to investment dollars
        firms = firms + firm_creation_rate * (I / firm_capital_cost)

        # 5) Update adoption
        A = A_next

    return {
        "year": yrs,
        "gdp": gdp,
        "employment": emp_rate,
        "consumption": cons,
        "investment": inv,
        "ai_share": ai_share,
    }

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Controls")

preset = st.sidebar.selectbox(
    "🎚️ Scenario Preset",
    [
        "Custom",
        "Baseline",
        "Fast AI + Slow Reskill (shock)",
        "Fast AI + Fast Reskill (adapt)",
        "Augment-first AI (less displacement)",
        "High Savings (more investment)",
    ],
)

defaults = {
    "Custom": {},
    "Baseline": dict(
        pop_m=5.0, firms_ratio=0.01,
        prod_sigma=0.45, auto_mean=0.55, auto_conc=10.0, init_unemp=0.05,
        ai_rate=0.04, replace_share=0.55, augment_strength=0.50, ai_prod_mult=1.8,
        reskill=0.35, reskill_effect=0.30, upskill_gain=0.08,
        save=0.20, dep=0.05, cap_boost=0.12,
        firm_create=0.06, firm_cost=2.0e9,
        years=30, sample=25000, seed=7
    ),
    "Fast AI + Slow Reskill (shock)": dict(
        pop_m=5.0, firms_ratio=0.01,
        prod_sigma=0.50, auto_mean=0.60, auto_conc=10.0, init_unemp=0.05,
        ai_rate=0.09, replace_share=0.75, augment_strength=0.25, ai_prod_mult=2.0,
        reskill=0.12, reskill_effect=0.20, upskill_gain=0.05,
        save=0.18, dep=0.06, cap_boost=0.10,
        firm_create=0.04, firm_cost=2.5e9,
        years=30, sample=25000, seed=7
    ),
    "Fast AI + Fast Reskill (adapt)": dict(
        pop_m=5.0, firms_ratio=0.01,
        prod_sigma=0.50, auto_mean=0.60, auto_conc=10.0, init_unemp=0.05,
        ai_rate=0.09, replace_share=0.70, augment_strength=0.35, ai_prod_mult=2.1,
        reskill=0.55, reskill_effect=0.40, upskill_gain=0.12,
        save=0.18, dep=0.06, cap_boost=0.10,
        firm_create=0.05, firm_cost=2.5e9,
        years=30, sample=25000, seed=7
    ),
    "Augment-first AI (less displacement)": dict(
        pop_m=5.0, firms_ratio=0.01,
        prod_sigma=0.45, auto_mean=0.55, auto_conc=10.0, init_unemp=0.05,
        ai_rate=0.06, replace_share=0.25, augment_strength=0.90, ai_prod_mult=1.5,
        reskill=0.30, reskill_effect=0.30, upskill_gain=0.07,
        save=0.20, dep=0.05, cap_boost=0.12,
        firm_create=0.06, firm_cost=2.0e9,
        years=30, sample=25000, seed=7
    ),
    "High Savings (more investment)": dict(
        pop_m=5.0, firms_ratio=0.01,
        prod_sigma=0.45, auto_mean=0.55, auto_conc=10.0, init_unemp=0.05,
        ai_rate=0.05, replace_share=0.55, augment_strength=0.50, ai_prod_mult=1.8,
        reskill=0.35, reskill_effect=0.30, upskill_gain=0.08,
        save=0.32, dep=0.05, cap_boost=0.18,
        firm_create=0.10, firm_cost=2.0e9,
        years=30, sample=25000, seed=7
    ),
}

D = defaults.get(preset, {})

with st.sidebar.expander("👥 Population + Firms", expanded=True):
    pop_m = st.slider("Population (millions)", 1.0, 10.0, float(D.get("pop_m", 5.0)), step=0.5)
    firms_ratio = st.slider("Firms as % of population", 0.1, 10.0, float(D.get("firms_ratio", 1.0))*100, step=0.1) / 100.0

with st.sidebar.expander("📊 Worker Distributions", expanded=True):
    prod_sigma = st.slider("Productivity dispersion (σ)", 0.10, 1.20, float(D.get("prod_sigma", 0.45)), step=0.05)
    auto_mean = st.slider("Avg automatability (0–1)", 0.05, 0.95, float(D.get("auto_mean", 0.55)), step=0.02)
    auto_conc = st.slider("Automatability concentration", 3.0, 30.0, float(D.get("auto_conc", 10.0)), step=1.0)
    init_unemp = st.slider("Initial unemployment rate", 0.00, 0.25, float(D.get("init_unemp", 0.05)), step=0.01)

with st.sidebar.expander("🤖 AI Adoption + Effects", expanded=True):
    ai_rate = st.slider("AI adoption rate per year", 0.01, 0.10, float(D.get("ai_rate", 0.04)), step=0.005)
    replace_share = st.slider("Share of automated tasks that REPLACE humans", 0.00, 1.00, float(D.get("replace_share", 0.55)), step=0.05)
    augment_strength = st.slider("Augmentation strength (boost to employed)", 0.00, 1.50, float(D.get("augment_strength", 0.50)), step=0.05)
    ai_prod_mult = st.slider("AI productivity multiplier on automated tasks", 0.50, 4.00, float(D.get("ai_prod_mult", 1.80)), step=0.10)

with st.sidebar.expander("🎓 Reskilling + Re-entry", expanded=True):
    reskill = st.slider("Re-employment (reskill) rate per year", 0.00, 1.00, float(D.get("reskill", 0.35)), step=0.05)
    reskill_effect = st.slider("Reskill reduces automatability by", 0.00, 0.80, float(D.get("reskill_effect", 0.30)), step=0.05)
    upskill_gain = st.slider("Productivity gain upon reskill", 0.00, 0.50, float(D.get("upskill_gain", 0.08)), step=0.02)

with st.sidebar.expander("💸 Macro Settings", expanded=True):
    save = st.slider("Savings rate (investment share of GDP)", 0.00, 0.50, float(D.get("save", 0.20)), step=0.02)
    dep = st.slider("Capital depreciation", 0.00, 0.15, float(D.get("dep", 0.05)), step=0.01)
    cap_boost = st.slider("Capital → TFP boost", 0.00, 0.40, float(D.get("cap_boost", 0.12)), step=0.02)
    firm_create = st.slider("Investment → firm creation rate", 0.00, 0.20, float(D.get("firm_create", 0.06)), step=0.01)
    firm_cost = st.number_input("Capital per new firm (USD)", value=float(D.get("firm_cost", 2.0e9)), step=1.0e8, format="%.0f")

with st.sidebar.expander("🧪 Simulation Controls", expanded=False):
    years = st.slider("Years to simulate", 5, 60, int(D.get("years", 30)), step=1)
    sample = st.slider("Sample workers (speed vs stability)", 5000, 60000, int(D.get("sample", 25000)), step=5000)
    seed = st.number_input("Random seed", value=int(D.get("seed", 7)), step=1)

# -----------------------------
# Run Simulation
# -----------------------------
out = simulate_economy(
    seed=int(seed),
    years=int(years),
    pop_millions=float(pop_m),
    firms_ratio=float(firms_ratio),
    prod_sigma=float(prod_sigma),
    auto_mean=float(auto_mean),
    auto_conc=float(auto_conc),
    init_unemp=float(init_unemp),
    ai_adoption_rate=float(ai_rate),
    replace_share=float(replace_share),
    augment_strength=float(augment_strength),
    ai_prod_mult=float(ai_prod_mult),
    reskill_rate=float(reskill),
    reskill_effect=float(reskill_effect),
    upskill_gain=float(upskill_gain),
    savings_rate=float(save),
    dep_rate=float(dep),
    capital_boost=float(cap_boost),
    firm_creation_rate=float(firm_create),
    firm_capital_cost=float(firm_cost),
    sample_workers=int(sample),
)

year = out["year"]
gdp = out["gdp"]
emp = out["employment"]
cons = out["consumption"]
inv = out["investment"]
ai_share = out["ai_share"]

# -----------------------------
# Layout
# -----------------------------
colA, colB = st.columns([2.2, 1.0], gap="large")

with colB:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("📌 Live Readout")

    st.metric("Population", f"{pop_m:.1f}M")
    st.metric("Firms (ratio)", f"{100*firms_ratio:.2f}% of population")
    st.metric("AI adoption rate", f"{100*ai_rate:.1f}% / year")
    st.metric("Replace share", f"{100*replace_share:.0f}%")
    st.metric("Reskill rate", f"{100*reskill:.0f}% / year")
    st.markdown("---")

    st.metric("GDP (final year)", f"${gdp[-1]/1e12:,.2f}T")
    st.metric("Employment (final year)", f"{100*emp[-1]:.1f}%")
    st.metric("Consumption (final year)", f"${cons[-1]/1e12:,.2f}T")
    st.metric("Investment (final year)", f"${inv[-1]/1e12:,.2f}T")
    st.metric("AI share of output (final year)", f"{100*ai_share[-1]:.1f}%")

    st.markdown("---")
    st.caption("Toy model for intuition. Not a forecast.")
    st.markdown("</div>", unsafe_allow_html=True)

with colA:
    tabs = st.tabs(["📈 GDP", "👷 Employment", "🧩 Components", "🧠 Model Notes"])

    with tabs[0]:
        fig = _nice_line(
            year,
            [gdp/1e12],
            ["GDP (T$)"],
            "GDP Over Time",
            "Trillions of USD"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        fig2 = _nice_line(
            year,
            [100*emp],
            ["Employment rate (%)"],
            "Employment Rate Over Time",
            "Percent"
        )
        fig2.update_yaxes(range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    with tabs[2]:
        fig3 = _nice_line(
            year,
            [cons/1e12, inv/1e12, 100*ai_share],
            ["Consumption (T$)", "Investment (T$)", "AI share (%)"],
            "Macro Components",
            "T$ (and % for AI share)"
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            """
            <div class="small">
            <b>Reading this:</b> Higher savings shifts GDP allocation from consumption → investment.
            Faster adoption raises AI output share, but if <i>replace_share</i> is high and <i>reskill</i> is low,
            employment can fall sharply.
            </div>
            """,
            unsafe_allow_html=True
        )

    with tabs[3]:
        st.markdown(
            """
### What this model is doing (plain English)

- **Workers** have (1) **productivity** and (2) **automatability**.
- **AI adoption** rises each year (bounded curve), and the **most-automatable tasks** get automated first.
- When tasks get automated:
  - Some are **replaced** (workers become unemployed) controlled by **Replace share**.
  - The rest are **augmented** (still employed, but productivity rises) controlled by **Augmentation strength**.
- **Reskilling** brings a fraction of unemployed workers back each year:
  - Their productivity increases (**Upskill gain**)
  - Their automatability falls (**Reskill reduces automatability**)

### Macro bookkeeping (kept intentionally simple)

Closed economy, no government, no trade:
- **GDP identity:** Y = C + I
- Choose savings rate s:
  - C = (1 - s)Y
  - I = sY
  - (So ex-post S = Y - C = I)

            """
        )

st.divider()
st.caption("Built with NumPy + Plotly + Streamlit.")
