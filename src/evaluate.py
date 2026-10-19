import pandas as pd, numpy as np, json
from sklearn.metrics import accuracy_score, confusion_matrix
from features import featurize
from decision_router import route
import joblib
ml = joblib.load("ml_model.joblib")
df = pd.read_csv("../results/ml_results.csv"); st = pd.read_csv("../data/stress_test.csv")
df["dm_flag"] = [route(i, m)[0] for i, m in zip(df.id, df.message)]
st["ml_pred"] = ml.predict(np.array([featurize(m) for m in st.message], dtype=float))
st["ml_p_sales"] = ml.predict_proba(np.array([featurize(m) for m in st.message], dtype=float))[:, 1].round(3)
st["dm_flag"] = [route(i, m)[0] for i, m in zip(st.id, st.message)]
st["support_score"] = [featurize(m)[0] for m in st.message]; st["sales_score"] = [featurize(m)[1] for m in st.message]
te = df[df.split == "test"]; mixed = df[df.case_type == "mixed"]
res = {}
for name, d in [("test20", te), ("mixed4", mixed), ("stress8", st)]:
    res[name] = {"ml_acc": accuracy_score(d.label, d.ml_pred), "dm_acc": accuracy_score(d.label, d.dm_flag),
                 "ml_cm": confusion_matrix(d.label, d.ml_pred, labels=[0, 1]).tolist(),
                 "dm_cm": confusion_matrix(d.label, d.dm_flag, labels=[0, 1]).tolist()}
print(json.dumps(res, indent=1))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(st[["id", "message", "label", "ml_pred", "ml_p_sales", "support_score", "sales_score", "dm_flag"]].to_string())
print(mixed[["id", "message", "label", "ml_pred", "ml_p_sales", "dm_flag", "split"]].to_string())
json.dump(res, open("../results/results.json", "w"), indent=1); df.to_csv("../results/all_results.csv", index=False); st.to_csv("../results/stress_results.csv", index=False)
