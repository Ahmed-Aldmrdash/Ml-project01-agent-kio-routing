# Routing for Agent KIO — CET351 Machine Learning, Lecture Project 1

Elsewedy University of Technology · Fall 2026

**Team:** Ahmed Mohamed Aldamardash · Moaaz Abd Aljawwad · Omar Ahmed Ramadan

Agent KIO is a customer chatbot with two specialist agents. This project builds the **intent router**
(message in → flag out: `0 = support`, `1 = sales`) and compares two ways to make that decision:

1. **Traditional ML classifier** — logistic regression with degree-2 polynomial features on two numeric
   features per message (`support_score`, `sales_score`).
2. **Decision model** — a model that reads the message plus routing instructions and returns
   `customer_service` or `sales` (Jev from TypeSafe AI, or a similar model; we ran Claude with the same instructions).

## Results

| Set | ML classifier | Decision model |
|---|---|---|
| Test set (20 messages) | 20/20 | 20/20 |
| Mixed intent (4 messages) | 2/4 | 4/4 |
| New hard messages (8) | 5/8 | 8/8 |

ML robustness over 200 random group-wise splits: 96% mean test accuracy (85% worst).
Recommendation: decision model as the router, the ML classifier as a free local fallback.

## Repository

```
data/      kio_routing_dataset.csv (60 synthetic messages, 40 train / 20 test by intent group, seed 42)
           stress_test.csv (8 extra hard messages)
src/       features.py, ml_router.py, decision_router.py, evaluate.py, figures.py, kio_demo.py
results/   decision_outputs.json, results.json, all_results.csv, stress_results.csv, fig/
docs/      ML_Project01_Agent_KIO_Routing.pptx (presentation), KIO_Project_Explained_AR.docx (Arabic notes)
```

## Run

```bash
pip install pandas scikit-learn matplotlib joblib
cd src
python ml_router.py    # train + evaluate the classifier, 200-split robustness check
python evaluate.py     # ML vs decision model -> results/results.json
python figures.py      # results/fig/boundary.png, accuracy.png
python kio_demo.py     # message -> router -> flag -> simulated support / sales agent
```

`decision_router.py` reads saved decision-model outputs by default. To call Jev instead:
`pip install typesafe-sdk`, set `TYPESAFE_API_KEY`, and use `provider="jev"`.

## Notes and limitations

- The 60 messages are synthetic, drafted with an AI assistant and reviewed by the team.
- The decision-model run is not a blind test: the same AI helped write the labels.
- Small sample: one error moves accuracy by 4–12 points.
