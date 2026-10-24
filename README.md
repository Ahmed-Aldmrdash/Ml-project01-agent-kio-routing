# Routing for Agent KIO — CET351 Machine Learning, Lecture Project 1

Elsewedy University of Technology · Fall 2026

**Team:** Ahmed Mohamed Aldamardash · Moaaz Abd Aljawwad · Omar Ahmed Ramadan

Agent KIO is a customer chatbot with two specialist agents. This project builds the **intent router**
(message in → flag out: `0 = support`, `1 = sales`) and compares two ways to make that decision:

1. **Traditional ML classifier** — logistic regression with degree-2 polynomial features on two numeric
   features per message (`support_score`, `sales_score`).
2. **Decision model** — `facebook/bart-large-mnli` (open-source, MIT licence) used as a zero-shot decision model:
   the message is compared with two written routing options and a wrapper returns a validated `customer_service` / `sales` route.

## Results (Google Colab run, 24 Oct 2026)

| Set | ML classifier | Decision model (BART-MNLI) |
|---|---|---|
| Test set (20 messages) | 20/20 | 16/20 |
| Mixed intent (4 messages) | 2/4 | 4/4 |
| New hard messages (8) | 5/8 | 5/8 |

ML robustness over 200 random group-wise splits: 96.4% mean test accuracy (85% worst).
Decision-model threshold t = 0.90, chosen on the 40 training messages only.
Recommendation: the classifier as the default router and the decision model only for messages with both problem and purchase cues (to be validated on fresh data).

## Repository

```
data/      kio_routing_dataset.csv (60 synthetic messages, 40 train / 20 test by intent group, seed 42)
           stress_test.csv (8 extra hard messages)
KIO_Routing.ipynb   full study: dataset, ML classifier, BART-MNLI decision model, comparison, KIO demo
src/       features.py, ml_router.py (stand-alone ML classifier scripts)
results/   colab_results.json, ml_results.csv, fig/boundary.png
docs/      ML_Project01_Agent_KIO_Routing.pptx (presentation), KIO_Project_Guide_AR.docx (team guide)
```

## Run

**Easiest:** open `KIO_Routing.ipynb` in Google Colab ([open in Colab](https://colab.research.google.com/github/Ahmed-Aldmrdash/ml-project01-agent-kio-routing/blob/main/KIO_Routing.ipynb)) and choose *Runtime → Run all*. It runs the full study, including the open-source decision model `facebook/bart-large-mnli`.

Local ML-only scripts:

```bash
pip install pandas scikit-learn matplotlib joblib
cd src && python ml_router.py
```

## Notes and limitations

- The 60 messages are synthetic, drafted with an AI assistant and reviewed by the team.
- The hybrid rule was designed after seeing the results and still needs a fresh test set.
- Small sample: one error moves accuracy by 4–12 points.
