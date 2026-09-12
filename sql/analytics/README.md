# SQL Analytics Layer

Six documented business questions, each as a standalone `.sql` file with the
question, query, and finding/insight written as comments in the file itself.
Findings marked "(example)" are placeholders illustrating the expected shape
of the result — replace them with real numbers after running against
loaded data (`python scripts/run_analytics.py` prints all six results).

| # | File | Business Question | SQL Technique |
|---|------|--------------------|----------------|
| 1 | [01_rolling_team_win_pct.sql](01_rolling_team_win_pct.sql) | What is each team's rolling 10-game win % trend? | Window function (`AVG() OVER ROWS BETWEEN`) |
| 2 | [02_player_efficiency_decline.sql](02_player_efficiency_decline.sql) | Which players are declining vs. their own season baseline? | `LAG`/rolling window comparison |
| 3 | [03_home_away_splits.sql](03_home_away_splits.sql) | How much does home court matter, per team? | Conditional aggregation (`FILTER`) |
| 4 | [04_back_to_back_fatigue.sql](04_back_to_back_fatigue.sql) | Does a back-to-back hurt shooting %? | Self-join (`LATERAL`) on schedule |
| 5 | [05_load_vs_performance.sql](05_load_vs_performance.sql) | Does player load predict next-game performance? | `LEAD` + `CORR()` + `NTILE` |
| 6 | [06_aging_curve.sql](06_aging_curve.sql) | What's the aging curve for veteran players? | `LAG` season-over-season deltas |

Run all six against the warehouse:

```bash
python scripts/run_analytics.py
```
