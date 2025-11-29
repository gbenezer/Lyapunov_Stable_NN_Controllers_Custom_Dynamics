from pathlib import Path
import pandas as pd
from scipy.stats import median_abs_deviation
import plotly.express as px

DATA_PATH = Path(
    "/home/benezer.gi/ondemand/dev/CS-7268-Verifiable-Machine-Learning/Lyapunov_Stable_NN_Controllers_Custom_Dynamics/Pendulum_Output_Metric_Data.csv"
)
df = pd.read_csv(DATA_PATH)
df["Metric"] = df["Metric"].astype("category")
df["System"] = df["System"].astype("category")
summary_df = df.groupby(["System", "Metric"])["Value"].agg(
    ["mean", "std", "median", lambda x: median_abs_deviation(x)]
)
# print(summary_df)
MAPPING = {}

for metric in pd.unique(df["Metric"]):
    current_df = df[df["Metric"] == metric]
    print(metric)
    fig = px.box(
        current_df,
        x="System",
        y="Value",
        color="System",
        # points="all",
        labels={"Value": metric},
    )
    fig.show()
