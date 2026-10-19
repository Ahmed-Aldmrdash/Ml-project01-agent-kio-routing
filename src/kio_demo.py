"""Agent KIO demo: message in -> router -> flag (0 support / 1 sales) -> simulated agent."""
import sys, joblib, pandas as pd
from features import featurize
from decision_router import route

def support_agent(message): print("    -> SUPPORT AGENT handles:", message)
def sales_agent(message):   print("    -> SALES AGENT handles:", message)
AGENTS = {0: support_agent, 1: sales_agent}

ml = joblib.load("ml_model.joblib")
def ml_route(message): return int(ml.predict([featurize(message)])[0])

if __name__ == "__main__":
    demo = pd.read_csv("../data/stress_test.csv")
    for r in demo.itertuples():
        print(f"\n[{r.id}] {r.message}")
        f_ml = ml_route(r.message); f_dm, raw, ok = route(r.id, r.message)
        print(f"  ML classifier flag = {f_ml} | decision model = {raw} -> flag {f_dm} (valid={ok}) | expected {r.label}")
        AGENTS[f_dm](r.message)
