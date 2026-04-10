from io import StringIO
import pandas as pd
import numpy as np

COEFFICIENTS = np.array([1.8173, 1.0997, -1.8583, 1.0631, 1.2922, 6.3804])

def predict(match_data, ball_by_ball_data):
    df = pd.read_csv(StringIO(ball_by_ball_data))
    df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0)
    df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0)
    if "wicket_type" not in df.columns:
        df["wicket_type"] = np.nan
    if "innings" in df.columns:
        df = df[df["innings"] == 1]
    df = df[df["ball"] < 3.0]
    runs_excl_extras = int(df["runs_off_bat"].sum())
    extras = int(df["extras"].sum())
    df["total_run"] = df["runs_off_bat"] + df["extras"]
    df["is_wicket"] = df["wicket_type"].notna()
    dots = int(((df["total_run"] == 0) & (~df["is_wicket"])).sum())
    fours = int((df["runs_off_bat"] == 4).sum())
    sixes = int((df["runs_off_bat"] == 6).sum())
    boundaries = fours + sixes
    wickets = int(df["wicket_type"].notna().sum())
    features = np.array([runs_excl_extras, dots, boundaries, np.exp(-wickets), extras, 1.0])
    predicted_runs = float(np.dot(COEFFICIENTS, features))
    return predicted_runs
