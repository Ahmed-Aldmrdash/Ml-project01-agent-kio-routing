import pandas as pd, numpy as np, joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from features import featurize

df = pd.read_csv("../data/kio_routing_dataset.csv")
tr, te = (df.split == "train").values, (df.split == "test").values
X = np.array([featurize(m) for m in df.message], dtype=float)
df["support_score"], df["sales_score"] = X[:, 0], X[:, 1]

model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False),
                      LogisticRegression(C=1.0, max_iter=1000))
model.fit(X[tr], df.label[tr])
df["ml_pred"] = model.predict(X)
df["ml_p_sales"] = model.predict_proba(X)[:, 1].round(3)

for name, m in [("train", tr), ("test", te)]:
    print(name, "accuracy", accuracy_score(df.label[m], df.ml_pred[m]))
print(confusion_matrix(df.label[te], df.ml_pred[te]))
print(classification_report(df.label[te], df.ml_pred[te], target_names=["support", "sales"], digits=3))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
print(df[te][["id", "message", "label", "ml_pred", "ml_p_sales", "support_score", "sales_score"]].to_string())
print("TRAIN errors:\n", df[tr & (df.label != df.ml_pred)][["id", "message", "label", "support_score", "sales_score"]].to_string())
lr = model[-1]; print("coef", lr.coef_.round(3), "intercept", lr.intercept_.round(3), model[1].get_feature_names_out(["sup", "sal"]))
df.to_csv("../results/ml_results.csv", index=False)
joblib.dump(model, "ml_model.joblib")

# Robustness: 200 random group-wise 40/20 splits (5 support + 5 sales families held out each time)
rng = np.random.default_rng(0); accs = []; per_case = {}
fams = {p: sorted({g for g in df.group_id if g.startswith(p)}) for p in "SL"}
for _ in range(200):
    test_g = set(rng.choice(fams["S"], 5, replace=False)) | set(rng.choice(fams["L"], 5, replace=False))
    m_te = df.group_id.isin(test_g).values
    mdl = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2, include_bias=False), LogisticRegression(C=1.0, max_iter=1000))
    mdl.fit(X[~m_te], df.label[~m_te]); p = mdl.predict(X[m_te])
    accs.append((p == df.label[m_te]).mean())
    for c, ok in zip(df.case_type[m_te], p == df.label[m_te]):
        per_case.setdefault(c, []).append(ok)
print(f"200 random group splits: mean test accuracy {np.mean(accs):.3f}, min {np.min(accs):.2f}, max {np.max(accs):.2f}")
for c, v in per_case.items(): print(f"  {c:35s} {np.mean(v):.2f}")
