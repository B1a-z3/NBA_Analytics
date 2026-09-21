"""
Streamlit dashboard: team trends, player risk flags, model accuracy over time,
and league-wide analytics (visualizing the sql/analytics/ business questions).

Reads static snapshots from dashboard/data/ (see scripts/export_dashboard_data.py).

Run:
    streamlit run dashboard/app.py
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# set_page_config() MUST be the first Streamlit command the script runs, in
# any code path -- even touching st.secrets before this (to check whether a
# secrets.toml exists) makes Streamlit render an internal notice first,
# which trips the "must be first" check just as visibly as an explicit
# st.* call would. So this goes before anything else.
st.set_page_config(page_title="NBA Analytics", layout="wide")

DATA_DIR = Path(__file__).resolve().parent / "data"

# Fixed-order categorical palette (colorblind-validated -- see the dataviz
# skill's references/palette.md). Assigned by identity (team, model name),
# never by rank, and never cycled/reordered per-chart.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# Single-hue sequential ramp (light -> dark blue) for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


@st.cache_data
def load_team_features():
    return _read("team_features")


@st.cache_data
def load_player_features():
    return _read("player_features")


@st.cache_data
def load_monitoring_log():
    return _read("monitoring_log")


@st.cache_data
def load_teams():
    return _read("teams")


@st.cache_data
def load_home_away_edge():
    """Per-team home-court edge (home win% minus away win%)."""
    return _read("home_away_edge")


@st.cache_data
def load_back_to_back_fatigue():
    """Team shooting % by rest bucket."""
    return _read("back_to_back_fatigue")


@st.cache_data
def load_load_vs_performance():
    """Load quartile vs next-game TS%."""
    return _read("load_vs_performance")


@st.cache_data
def load_aging_curve():
    return _read("aging_curve")


@st.cache_data
def load_decline_feature_importances():
    path = DATA_DIR / "decline_feature_importances.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path).sort_values("importance", ascending=True)


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
        st.info("No team feature data yet -- snapshot data missing.")
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
        st.info("No player feature data yet -- snapshot data missing.")
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
        st.info("No monitoring data in the snapshot.")
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
            st.info("No feature-importance snapshot found.")
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
        st.info("No aging-curve data in the snapshot.")
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
