"""
Streamlit dashboard: team trends, player risk flags, model accuracy over time,
and league-wide analytics (visualizing the sql/analytics/ business questions).

Run:
    streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# set_page_config() MUST be the first Streamlit command the script runs, in
# any code path -- even touching st.secrets before this (to check whether a
# secrets.toml exists) makes Streamlit render an internal notice first,
# which trips the "must be first" check just as visibly as an explicit
# st.* call would. So this goes before anything else.
st.set_page_config(page_title="NBA Analytics", layout="wide")

# Streamlit Community Cloud's secrets manager (Settings -> Secrets, TOML
# format) does NOT automatically become an OS environment variable -- it's
# only exposed via st.secrets. config/db.py reads DATABASE_URL via
# os.getenv() at import time, so bridge it here BEFORE that import runs.
# Locally this is a no-op: st.secrets raises/returns empty when no
# secrets.toml exists, and .env (via python-dotenv in config/db.py) covers
# local development instead.
try:
    if "DATABASE_URL" in st.secrets:
        os.environ["DATABASE_URL"] = st.secrets["DATABASE_URL"]
except FileNotFoundError:
    pass  # no secrets.toml -- local dev, config/db.py falls back to .env

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

engine = get_engine()
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"

# Fixed-order categorical palette (colorblind-validated -- see the dataviz
# skill's references/palette.md). Assigned by identity (team, model name),
# never by rank, and never cycled/reordered per-chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Single-hue sequential ramp (light -> dark blue) for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]


@st.cache_data(ttl=600)
def load_team_features():
    return pd.read_sql("SELECT * FROM team_game_features", engine)


@st.cache_data(ttl=600)
def load_player_features():
    return pd.read_sql("SELECT * FROM player_game_features", engine)


@st.cache_data(ttl=600)
def load_monitoring_log():
    return pd.read_sql("""
        SELECT mv.model_name, mv.version_tag, mml.week_start, mml.n_predictions,
               mml.accuracy, mml.log_loss, mml.brier_score
        FROM model_monitoring_log mml
        JOIN model_versions mv ON mv.model_version_id = mml.model_version_id
        ORDER BY mml.week_start
    """, engine)


@st.cache_data(ttl=600)
def load_teams():
    return pd.read_sql("SELECT team_id, abbreviation, name FROM teams", engine)


@st.cache_data(ttl=600)
def load_home_away_edge():
    """Per-team home-court edge (home win% minus away win%) -- the chart
    version of sql/analytics/03_home_away_splits.sql."""
    return pd.read_sql("""
        WITH home_results AS (
            SELECT home_team_id AS team_id, COUNT(*) home_games,
                   COUNT(*) FILTER (WHERE home_score > away_score) home_wins
            FROM games WHERE home_score IS NOT NULL GROUP BY home_team_id
        ),
        away_results AS (
            SELECT away_team_id AS team_id, COUNT(*) away_games,
                   COUNT(*) FILTER (WHERE away_score > home_score) away_wins
            FROM games WHERE home_score IS NOT NULL GROUP BY away_team_id
        )
        SELECT t.abbreviation,
               ROUND((h.home_wins::numeric / h.home_games
                      - a.away_wins::numeric / a.away_games), 3) AS home_court_edge
        FROM teams t
        JOIN home_results h ON h.team_id = t.team_id
        JOIN away_results a ON a.team_id = t.team_id
        ORDER BY home_court_edge DESC
    """, engine)


@st.cache_data(ttl=600)
def load_back_to_back_fatigue():
    """Team shooting % by rest bucket -- the chart version of
    sql/analytics/04_back_to_back_fatigue.sql. Reads a precomputed summary
    table (features/build_dashboard_summaries.py) rather than joining the
    324MB+ raw player_game_stats table live: keeps the dashboard fast and
    keeps that huge table out of any cloud-hosted copy of the database
    entirely (a free-tier Postgres host typically caps around 500MB)."""
    return pd.read_sql("SELECT * FROM dashboard_back_to_back_fatigue ORDER BY sort_key", engine)


@st.cache_data(ttl=600)
def load_load_vs_performance():
    """Load quartile vs next-game TS% -- the chart version of
    sql/analytics/05_load_vs_performance.sql, built on the already-
    materialized player_game_features (a single windowed pass, not a
    from-scratch rebuild)."""
    return pd.read_sql("""
        WITH ordered AS (
            SELECT player_id, game_date, load_index, ts_pct,
                   LEAD(ts_pct) OVER (PARTITION BY player_id ORDER BY game_date) AS next_ts_pct,
                   NTILE(4) OVER (ORDER BY load_index) AS load_quartile
            FROM player_game_features
            WHERE load_index IS NOT NULL
        )
        SELECT load_quartile, COUNT(*) AS n,
               ROUND(AVG(next_ts_pct)::numeric, 3) AS avg_next_game_ts_pct
        FROM ordered
        WHERE next_ts_pct IS NOT NULL
        GROUP BY load_quartile
        ORDER BY load_quartile
    """, engine)


@st.cache_data(ttl=600)
def load_aging_curve():
    """Chart version of sql/analytics/06_aging_curve.sql. Reads a
    precomputed summary table (see load_back_to_back_fatigue's docstring
    for why) instead of joining player_game_stats + games + players live."""
    return pd.read_sql("SELECT * FROM dashboard_aging_curve ORDER BY age_at_season_start", engine)


@st.cache_data(ttl=600)
def load_decline_feature_importances():
    path = MODEL_DIR / "player_decline.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    model, features = bundle["model"], bundle["features"]
    return pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)


st.title("NBA Player Performance & Game Outcome Prediction System")

tab_team, tab_player, tab_monitoring, tab_league = st.tabs(
    ["Team Trends", "Player Risk Flags", "Model Accuracy Over Time", "League Analytics"]
)

# ---------------------------------------------------------------------------
with tab_team:
    st.subheader("Team Form Trends")
    team_features = load_team_features()
    teams = load_teams()

    if team_features.empty:
        st.info("No team feature data yet -- run ingestion + features/build_features.py.")
    else:
        team_features = team_features.merge(teams, on="team_id", how="left")
        team_choice = st.selectbox(
            "Team", sorted(team_features["abbreviation"].dropna().unique())
        )
        team_df = team_features[team_features["abbreviation"] == team_choice].sort_values("game_date")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(team_df, x="game_date", y="rolling_win_pct",
                          title=f"{team_choice} — Rolling 10-Game Win %")
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.line(team_df, x="game_date", y="rolling_point_diff",
                           title=f"{team_choice} — Rolling Point Differential")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Back-to-back rate this season**")
        st.metric("Back-to-back games", int(team_df["is_back_to_back"].sum()))

        st.divider()
        st.markdown("**Compare rolling form across teams**")
        compare_teams = st.multiselect(
            "Teams to compare (pick 2-8)",
            sorted(team_features["abbreviation"].dropna().unique()),
            default=sorted(team_features["abbreviation"].dropna().unique())[:3],
            max_selections=8,
        )
        if len(compare_teams) >= 2:
            compare_df = team_features[team_features["abbreviation"].isin(compare_teams)]
            fig_cmp = px.line(
                compare_df, x="game_date", y="rolling_win_pct", color="abbreviation",
                category_orders={"abbreviation": compare_teams},
                color_discrete_sequence=CATEGORICAL,
                title="Rolling 10-Game Win % — Team Comparison",
                labels={"abbreviation": "Team", "rolling_win_pct": "Rolling win %", "game_date": "Date"},
            )
            fig_cmp.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.caption("Pick at least 2 teams to compare.")

        st.markdown("**Home-court edge, all teams** (home win% − away win%)")
        edge_df = load_home_away_edge()
        fig_edge = px.bar(
            edge_df, x="home_court_edge", y="abbreviation", orientation="h",
            color="home_court_edge", color_continuous_scale=SEQUENTIAL_BLUE,
            labels={"home_court_edge": "Home-court edge", "abbreviation": "Team"},
        )
        fig_edge.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig_edge, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_player:
    st.subheader("Player Decline Risk Flags")
    player_features = load_player_features()

    if player_features.empty:
        st.info("No player feature data yet -- run ingestion + features/build_features.py.")
    else:
        player_features["decline_gap"] = (
            player_features["season_to_date_ts_pct"] - player_features["rolling_ts_pct"]
        )
        latest = (
            player_features.sort_values("game_date")
            .groupby("player_id").tail(1)
            .sort_values("decline_gap", ascending=False)
        )
        flagged = latest[latest["decline_gap"] > 0.05]

        st.markdown(f"**{len(flagged)} players currently flagged for efficiency decline** "
                    f"(rolling 5-game TS% more than 5 points below season baseline)")
        st.dataframe(
            flagged[["full_name", "rolling_ts_pct", "season_to_date_ts_pct",
                     "decline_gap", "rolling_load_index", "age_years"]]
            .rename(columns={
                "full_name": "Player", "rolling_ts_pct": "Rolling TS%",
                "season_to_date_ts_pct": "Season TS%", "decline_gap": "Decline Gap",
                "rolling_load_index": "Load Index", "age_years": "Age",
            }).round(3),
            use_container_width=True,
        )

        player_choice = st.selectbox("Inspect player trend", sorted(player_features["full_name"].dropna().unique()))
        p_df = player_features[player_features["full_name"] == player_choice].sort_values("game_date")
        fig = px.line(p_df, x="game_date", y=["rolling_ts_pct", "season_to_date_ts_pct"],
                      color_discrete_sequence=CATEGORICAL,
                      title=f"{player_choice} — Rolling vs Season TS%")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("**Decline risk landscape** — every player's latest game, "
                     "age vs. decline gap, colored by recent workload")
        scatter_df = latest.dropna(subset=["age_years", "decline_gap", "rolling_load_index"])
        fig_scatter = px.scatter(
            scatter_df, x="age_years", y="decline_gap", color="rolling_load_index",
            color_continuous_scale=SEQUENTIAL_BLUE, hover_name="full_name",
            labels={"age_years": "Age", "decline_gap": "Decline gap (TS%)",
                    "rolling_load_index": "Load index"},
        )
        fig_scatter.add_hline(y=0.05, line_dash="dash", line_color="#898781",
                               annotation_text="flag threshold", annotation_position="top left")
        st.plotly_chart(fig_scatter, use_container_width=True)

        col_hist, col_note = st.columns([2, 1])
        with col_hist:
            fig_hist = px.histogram(
                latest.dropna(subset=["decline_gap"]), x="decline_gap", nbins=40,
                color_discrete_sequence=[CATEGORICAL[0]],
                labels={"decline_gap": "Decline gap (TS%)"},
                title="Distribution of decline gap across all players",
            )
            fig_hist.add_vline(x=0.05, line_dash="dash", line_color="#898781")
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_note:
            st.caption(
                "Decline gap = season-to-date TS% minus rolling 5-game TS%. "
                "Positive means recent play is below the player's own season "
                "baseline; the dashed line marks the 0.05 flag threshold used "
                "in the table above and in sql/analytics/02_player_efficiency_decline.sql."
            )

# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Model Accuracy Over Time (Drift Monitoring)")
    log_df = load_monitoring_log()

    if log_df.empty:
        st.info("No monitoring data yet -- run monitoring/update_drift_log.py after some predictions "
                "have been logged and their games completed.")
    else:
        all_models = sorted(log_df["model_name"].unique())
        st.markdown("**All models, weekly accuracy — compared**")
        fig_all = px.line(
            log_df.sort_values("week_start"), x="week_start", y="accuracy", color="model_name",
            markers=True, category_orders={"model_name": all_models},
            color_discrete_sequence=CATEGORICAL,
            labels={"model_name": "Model", "accuracy": "Accuracy", "week_start": "Week"},
        )
        fig_all.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_all, use_container_width=True)

        st.divider()
        model_choice = st.selectbox("Inspect one model", all_models)
        m_df = log_df[log_df["model_name"] == model_choice].sort_values("week_start")

        fig = px.line(m_df, x="week_start", y="accuracy", markers=True,
                      color_discrete_sequence=[CATEGORICAL[0]],
                      title=f"{model_choice} — Weekly Accuracy")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.line(m_df, x="week_start", y="log_loss", markers=True,
                            color_discrete_sequence=[CATEGORICAL[0]], title="Log Loss")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.line(m_df, x="week_start", y="brier_score", markers=True,
                            color_discrete_sequence=[CATEGORICAL[0]], title="Brier Score")
            st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(m_df, use_container_width=True)

        st.divider()
        st.markdown("**Player decline model — feature importances**")
        importances_df = load_decline_feature_importances()
        if importances_df is None:
            st.info("No trained player_decline model found -- run models/train_player_decline.py.")
        else:
            fig_imp = px.bar(
                importances_df, x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale=SEQUENTIAL_BLUE,
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_league:
    st.subheader("League-Wide Analytics")
    #st.caption("Interactive versions of the six documented business questions "
      #         "in sql/analytics/ — see that folder for the full write-up of "
      #         "each finding.")

    metric_choice = st.radio("Aging curve metric", ["Points per game", "True Shooting %"], horizontal=True)
    aging_df = load_aging_curve()
    if aging_df.empty:
        st.info("No aging-curve data yet -- run ingestion/enrich_player_bio.py to fill in player birth dates.")
    else:
        y_col = "league_avg_pts_at_age" if metric_choice == "Points per game" else "league_avg_ts_pct_at_age"
        fig_age = px.line(
            aging_df, x="age_at_season_start", y=y_col, markers=True,
            color_discrete_sequence=[CATEGORICAL[0]],
            labels={"age_at_season_start": "Age at season start", y_col: metric_choice},
            title=f"League-average {metric_choice} by age",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    col_b2b, col_load = st.columns(2)
    with col_b2b:
        st.markdown("**Back-to-back fatigue**")
        b2b_df = load_back_to_back_fatigue()
        fig_b2b = px.bar(
            b2b_df, x="rest_bucket", y="avg_team_fg_pct",
            color="rest_bucket", color_discrete_sequence=CATEGORICAL,
            labels={"rest_bucket": "Rest before game", "avg_team_fg_pct": "Avg team FG%"},
        )
        fig_b2b.update_layout(showlegend=False)
        fig_b2b.update_yaxes(range=[0.4, 0.48])
        st.plotly_chart(fig_b2b, use_container_width=True)
    with col_load:
        st.markdown("**Load quartile vs. next-game efficiency**")
        load_df = load_load_vs_performance()
        load_df["load_quartile"] = load_df["load_quartile"].map(
            {1: "Q1 (lowest)", 2: "Q2", 3: "Q3", 4: "Q4 (highest)"}
        )
        fig_load = px.bar(
            load_df, x="load_quartile", y="avg_next_game_ts_pct",
            color="load_quartile", color_discrete_sequence=CATEGORICAL,
            labels={"load_quartile": "Load quartile", "avg_next_game_ts_pct": "Avg next-game TS%"},
        )
        fig_load.update_layout(showlegend=False)
        st.plotly_chart(fig_load, use_container_width=True)
