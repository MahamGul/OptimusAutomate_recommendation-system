import streamlit as st
import pandas as pd
from recommender import (
    load_data, build_content_model, build_collab_model,
    recommend_content, recommend_collab, hybrid_recommend
)
from llm import get_llm_explanation, get_movie_vibe

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BingeWatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS  — BingeWatch AI theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=Bebas+Neue&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #000000 !important;
    color: #DDDDDD !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    border-right: 1px solid #1a1a1a !important;
}

/* ── Hero header ── */
.hero {
    padding: 2.5rem 0 1.5rem 0;
    text-align: center;
}
.hero-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    letter-spacing: 0.06em;
    line-height: 1;
    background: linear-gradient(135deg, #CB2957 0%, #ff6b9d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 500;
}
.hero-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #CB2957, transparent);
    margin: 1.5rem auto;
    width: 60%;
    border: none;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #CCCCCC !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3 {
    color: #EEEEEE !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.08em !important;
}
.sidebar-section {
    background: #141414;
    border-radius: 10px;
    padding: 1rem 1rem 0.5rem 1rem;
    margin-bottom: 1rem;
    border: 1px solid #1e1e1e;
}

/* ── Selectbox / Input ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input {
    background-color: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #EEEEEE !important;
}
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #CB2957 !important;
}

/* ── Primary button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #CB2957 0%, #a01f44 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(203, 41, 87, 0.45) !important;
}

/* ── Radio pills ── */
[data-testid="stRadio"] div[role="radiogroup"] {
    flex-direction: row !important;
    gap: 0.5rem !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 20px;
    padding: 0.35rem 1rem !important;
    font-size: 0.82rem !important;
    color: #BBB !important;
    cursor: pointer;
    transition: all 0.15s;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #CB2957 !important;
    border-color: #CB2957 !important;
    color: #fff !important;
}

/* ── Slider ── */
[data-testid="stSlider"] {
    padding: 0.25rem 0.5rem !important;
}
[data-testid="stSlider"] div[role="slider"] {
    background-color: #CB2957 !important;
    border: 3px solid #ffffff !important;
    width: 18px !important;
    height: 18px !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:first-child {
    background-color: #2a2a2a !important;
}
[data-testid="stSlider"] [data-testid="stSliderTrack"] > div:nth-child(2) {
    background-color: #CB2957 !important;
}
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {
    color: #EEEEEE !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    background: transparent !important;
}

/* ── Movie card ── */
.movie-card {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s, transform 0.15s;
}
.movie-card:hover {
    border-color: #CB2957;
    transform: translateX(3px);
}
.movie-rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: #CB2957;
    min-width: 2.5rem;
    line-height: 1;
}
.movie-info { flex: 1; }
.movie-title {
    font-weight: 600;
    font-size: 1rem;
    color: #EEEEEE;
    margin-bottom: 0.15rem;
}
.movie-genres {
    font-size: 0.78rem;
    color: #666;
    margin-bottom: 0.2rem;
}
.movie-score-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 0.5rem;
    background: linear-gradient(90deg, #CB2957, #ff6b9d);
}

/* ── Vibe tag ── */
.vibe-box {
    background: #120008;
    border: 1px solid #3a0020;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    color: #e87ca0;
    font-size: 0.88rem;
    font-style: italic;
    margin: 0.75rem 0 1.25rem 0;
    text-align: center;
}

/* ── AI explanation card ── */
.ai-card {
    background: linear-gradient(135deg, #0d0007 0%, #100010 100%);
    border: 1px solid #CB2957;
    border-radius: 12px;
    padding: 1.5rem 1.75rem;
    margin-top: 1.5rem;
    position: relative;
}
.ai-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    color: #CB2957;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.ai-text {
    color: #CCCCCC;
    line-height: 1.75;
    font-size: 0.95rem;
}

/* ── Selected movie banner ── */
.selected-banner {
    background: #0d0d0d;
    border-left: 3px solid #CB2957;
    padding: 0.85rem 1.25rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 1.5rem;
}
.selected-label {
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #CB2957;
    font-weight: 600;
    margin-bottom: 0.15rem;
}
.selected-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #EEEEEE;
}

/* ── Stats strip ── */
.stats-strip {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.stat-pill {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-size: 0.82rem;
    color: #888;
}
.stat-pill span {
    color: #CB2957;
    font-weight: 700;
}

/* ── Section titles ── */
.section-heading {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.5rem;
    letter-spacing: 0.08em;
    color: #EEEEEE;
    margin-bottom: 0.1rem;
}
.section-sub {
    font-size: 0.8rem;
    color: #555;
    margin-bottom: 1.25rem;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #444;
}
.empty-state .big-icon {
    font-size: 3rem;
    margin-bottom: 0.75rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 3px; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & CACHE MODELS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_models():
    movies, ratings = load_data()
    content_sim, content_idx, movies_df = build_content_model(movies)
    collab_sim, collab_idx, matrix = build_collab_model(ratings, movies)
    return movies, content_sim, content_idx, movies_df, collab_sim, collab_idx, matrix


with st.spinner("Loading models..."):
    movies, content_sim, content_idx, movies_df, collab_sim, collab_idx, matrix = get_models()


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-logo">BingeWatch AI</div>
    <div class="hero-sub">Intelligent Movie Recommendations · Powered by ML + LLM</div>
    <hr class="hero-divider">
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Brand mark ──
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.1em;
                    background:linear-gradient(135deg,#CB2957,#ff6b9d);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">
            🎬 BINGE<span style="-webkit-text-fill-color:#fff;">WATCH</span>
        </div>
        <div style="font-size:0.65rem; color:#444; letter-spacing:0.2em; text-transform:uppercase; margin-top:0.1rem;">
            AI · ML · LLM Powered
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #1a1a1a; margin: 0.5rem 0 1.2rem 0;">
    """, unsafe_allow_html=True)

    # ── Algorithm ──
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#CB2957;font-weight:700;margin-bottom:0.4rem;">⚙ Algorithm</p>', unsafe_allow_html=True)
    model_type = st.radio(
        "Algorithm",
        ["Content-Based", "Collaborative", "Hybrid ✨"],
        label_visibility="collapsed"
    )

    st.markdown('<hr style="border:none;border-top:1px solid #161616;margin:1rem 0;">', unsafe_allow_html=True)

    # ── Results count ──
    st.markdown('<p style="font-size:0.7rem;letter-spacing:0.18em;text-transform:uppercase;color:#CB2957;font-weight:700;margin-bottom:0.4rem;">🎯 How Many Results</p>', unsafe_allow_html=True)
    top_n = st.slider("Number of recommendations", 1, 15, 8, label_visibility="collapsed")

    st.markdown('<hr style="border:none;border-top:1px solid #161616;margin:1rem 0;">', unsafe_allow_html=True)

    # ── Algorithm explainer cards ──
    algo_info = {
        "Content-Based": ("🎭", "Genre Match", "Finds movies sharing the same genres and release era as your pick."),
        "Collaborative": ("👥", "User Taste", "Recommends what people with similar ratings history also loved."),
        "Hybrid ✨":     ("⚡", "Best of Both", "Blends genre similarity + user behaviour for the smartest picks."),
    }
    icon, label, desc = algo_info[model_type]
    st.markdown(f"""
    <div style="background:#0d0d0d; border:1px solid #1e1e1e; border-left: 3px solid #CB2957;
                border-radius:10px; padding:0.9rem 1rem; margin-bottom:1rem;">
        <div style="font-size:1.4rem; margin-bottom:0.3rem;">{icon}</div>
        <div style="font-size:0.8rem; font-weight:700; color:#EEEEEE; margin-bottom:0.25rem;">{label}</div>
        <div style="font-size:0.75rem; color:#666; line-height:1.5;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Live stats ──
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; gap:0.4rem;">
        <div style="background:#0a0a0a; border:1px solid #181818; border-radius:8px;
                    padding:0.55rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; color:#555;">Movies indexed</span>
            <span style="font-size:0.85rem; font-weight:700; color:#CB2957;">{len(movies):,}</span>
        </div>
        <div style="background:#0a0a0a; border:1px solid #181818; border-radius:8px;
                    padding:0.55rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; color:#555;">Showing top</span>
            <span style="font-size:0.85rem; font-weight:700; color:#CB2957;">{top_n}</span>
        </div>
        <div style="background:#0a0a0a; border:1px solid #181818; border-radius:8px;
                    padding:0.55rem 0.9rem; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:0.75rem; color:#555;">Engine</span>
            <span style="font-size:0.75rem; font-weight:700; color:#CB2957;">Ollama · llama3</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div style="margin-top:1.5rem; text-align:center;">
        <div style="font-size:0.65rem; color:#2a2a2a; letter-spacing:0.1em;">
            BUILT WITH STREAMLIT · SKLEARN · OLLAMA
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN — MOVIE SELECTION
# ─────────────────────────────────────────────
movie_list = sorted(movies['title'].dropna().unique().tolist())

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    selected_movie = st.selectbox(
        "Search for a movie",
        movie_list,
        placeholder="Start typing a movie title...",
        label_visibility="collapsed"
    )
with col_btn:
    run = st.button("Find Matches →")


# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if not run:
    st.markdown("""
    <div class="empty-state">
        <div class="big-icon">🍿</div>
        <div style="font-size:1.1rem; color:#555; margin-bottom:0.5rem;">Pick a movie above and hit <b>Find Matches</b></div>
        <div style="font-size:0.85rem; color:#3a3a3a;">We'll use ML + AI to find what you should watch next</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Get selected movie genres ──
selected_genres = ""
if selected_movie in content_idx:
    row = movies_df.iloc[content_idx[selected_movie]]
    selected_genres = row['genres'].replace(' ', ', ')

# ── Selected banner ──
st.markdown(f"""
<div class="selected-banner">
    <div class="selected-label">Now Matching</div>
    <div class="selected-title">{selected_movie}</div>
    {f'<div style="font-size:0.8rem;color:#555;margin-top:0.25rem;">{selected_genres}</div>' if selected_genres else ''}
</div>
""", unsafe_allow_html=True)

# ── Vibe check (AI) ──
if selected_genres:
    with st.spinner("Getting vibe check..."):
        vibe = get_movie_vibe(selected_movie, selected_genres)
    if vibe:
        st.markdown(f'<div class="vibe-box">✨ {vibe}</div>', unsafe_allow_html=True)

# ── Run recommendation ──
with st.spinner("Finding your next binge..."):
    if model_type == "Content-Based":
        results = recommend_content(selected_movie, content_sim, content_idx, movies_df, top_n)
    elif model_type == "Collaborative":
        results = recommend_collab(selected_movie, collab_sim, collab_idx, matrix, top_n)
    else:
        results = hybrid_recommend(
            selected_movie, content_sim, content_idx, movies_df,
            collab_sim, collab_idx, matrix, top_n
        )

# ── Stats strip ──
algo_label = {"Content-Based": "Genre & Feature Matching",
              "Collaborative": "User Behavior Patterns",
              "Hybrid ✨": "Content + Collaborative Blend"}[model_type]

st.markdown(f"""
<div class="stats-strip">
    <div class="stat-pill">Found <span>{len(results)}</span> matches</div>
    <div class="stat-pill">Method: <span>{algo_label}</span></div>
    <div class="stat-pill">Dataset: <span>{len(movies):,}</span> movies</div>
</div>
""", unsafe_allow_html=True)

# ── Movie cards ──
st.markdown('<div class="section-heading">Your Recommendations</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Ranked by similarity score</div>', unsafe_allow_html=True)

if not results:
    st.warning(f"No recommendations found for **{selected_movie}** with this algorithm. Try switching to Hybrid mode.")
else:
    for i, movie in enumerate(results, 1):
        bar_width = int(movie["score"] * 100)

        # For collab results, look up genres from movies_df
        genres = movie.get("genres", "")
        if not genres and movie["title"] in content_idx:
            genres = movies_df.iloc[content_idx[movie["title"]]]["genres"].replace(" ", ", ")

        title = movie["title"]
        score = movie["score"]
        bar_w = max(bar_width, 5)

        card_html = f"""<div class="movie-card">
            <div class="movie-rank">#{i:02d}</div>
            <div class="movie-info">
                <div class="movie-title">{title}</div>"""

        if genres:
            card_html += f'<div class="movie-genres">{genres}</div>'

        card_html += f"""<div class="movie-score-bar" style="width:{bar_w}%;"></div>
            </div>
            <div style="color:#555; font-size:0.82rem; text-align:right; min-width:3.5rem;">
                <span style="color:#CB2957; font-weight:700; font-size:1rem;">{score:.3f}</span><br>
                <span>similarity</span>
            </div>
        </div>"""

        st.markdown(card_html, unsafe_allow_html=True)

    # ── AI Explanation ──
    with st.spinner("Binge Bot is cooking up an explanation... 🤖"):
        explanation = get_llm_explanation(selected_movie, results, model_type)

    st.markdown(f"""
    <div class="ai-card">
        <div class="ai-badge">🤖 Binge Bot Says</div>
        <div class="ai-text">{explanation}</div>
    </div>
    """, unsafe_allow_html=True)