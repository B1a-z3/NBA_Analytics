"""
Streamlit dashboard: team trends, player risk flags, model accuracy over time.

Run:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.db import get_engine  # noqa: E402

st.set_page_config(page_title="NBA Analytics", layout="wide")
engine = get_engine()


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


st.title("NBA Player Performance & Game Outcome Prediction System")

tab_team, tab_player, tab_monitoring = st.tabs(
    ["Team Trends", "Player Risk Flags", "Model Accuracy Over Time"]
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
                      title=f"{player_choice} — Rolling vs Season TS%")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("Model Accuracy Over Time (Drift Monitoring)")
    log_df = load_monitoring_log()

    if log_df.empty:
        st.info("No monitoring data yet -- run monitoring/update_drift_log.py after some predictions "
                "have been logged and their games completed.")
    else:
        model_choice = st.selectbox("Model", sorted(log_df["model_name"].unique()))
        m_df = log_df[log_df["model_name"] == model_choice].sort_values("week_start")

        fig = px.line(m_df, x="week_start", y="accuracy", markers=True,
                      title=f"{model_choice} — Weekly Accuracy")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.line(m_df, x="week_start", y="log_loss", markers=True, title="Log Loss")
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3 = px.line(m_df, x="week_start", y="brier_score", markers=True, title="Brier Score")
            st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(m_df, use_container_width=True)
