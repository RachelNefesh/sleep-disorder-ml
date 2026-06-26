# Sleep Disorders & Sleep Quality — A Machine Learning Project

This is an end-to-end machine learning project where I dig into the Sleep Health
and Lifestyle dataset to answer two questions:

1. **What health and lifestyle factors best predict whether someone has a sleep disorder?**
2. **What factors are linked to better sleep quality?**

I wanted to do more than just fit a model and report a high number, so a lot of
the project is about doing the methodology carefully — avoiding data leakage,
checking results with cross-validation instead of one lucky train/test split, and
being honest about *why* a model scored well.

## The data

- **Source:** [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (Kaggle, by Laksika Tharmalingam)
- **Size:** 374 people, 13 features
- **What's in it:** age, gender, occupation, sleep duration, sleep quality,
  stress level, BMI category, blood pressure, heart rate, daily steps, and whether
  the person has a sleep disorder (None / Insomnia / Sleep Apnea)

The CSV is included in this repo as `Sleep_health_and_lifestyle_dataset.csv`,
so no separate download is needed to run the notebook.

## What I did

**Cleaning and feature engineering.** I one-hot encoded the categorical columns
(gender, occupation, BMI, disorder) and wrote a custom encoder for blood pressure,
which comes in as a string like `120/80` — I split it into systolic and diastolic
numbers and also binned it into clinical levels (Low / Normal / Elevated / High).

**Avoiding data leakage.** I scale the numeric features inside scikit-learn
`Pipeline`s rather than scaling the whole dataset up front. That way the scaler
only ever learns from the training data, so nothing about the test set leaks into
the model and inflates the score. (The tree models don't get scaled at all, since
they split on order, not size.)

**Question 1 — predicting a sleep disorder.** I used a LASSO logistic regression
with permutation importance to rank features, ran single-variable logistic
regressions on the most clinically relevant predictors to see how strong each one
is on its own, added a Random Forest, and then compared four models
(Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) using 5-fold
cross-validation. I evaluated with ROC-AUC and confusion matrices.

**Question 2 — predicting sleep quality.** I used linear regression for the
continuous predictors and logistic regression for a yes/no "good sleep" version of
the target.

## What I found

**Blood pressure is the strongest single predictor of a disorder** — systolic
alone hits 0.91 accuracy. But I dug into *why*, and it's more interesting than the
number: blood pressure is near-deterministic at the extremes (95% of Normal-BP
people are healthy, 93% of High-BP people have a disorder), but the biggest group,
Elevated, makes up about 62% of the data and is genuinely mixed. The continuous
blood-pressure value does better (0.91) than just using the category (about 0.82),
because it captures the gradient inside that ambiguous middle band. The high
accuracy partly reflects how tightly blood pressure tracks the label in this
dataset, not just model skill — worth being upfront about.

**The simplest model won, and that surprised me at first.** On a single train/test
split the Random Forest looked best (0.93). But under 5-fold cross-validation,
Logistic Regression was both the most accurate and the most stable (0.88), while
the Random Forest dropped to 0.70 with big swings across folds. The lesson: that
single split was optimistic, and on a small dataset a simpler model often
generalizes better than a complex ensemble that overfits. This is the part of the
project I learned the most from.

**For sleep quality**, lower stress was the strongest predictor (R² = 0.83),
followed by longer sleep duration (R² = 0.77), with lower heart rate and more
physical activity also linked to better sleep.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

Techniques: feature engineering, custom encoding, leakage-safe pipelines, LASSO,
permutation importance, cross-validation, ROC-AUC, confusion matrices, logistic and
linear regression, ensemble models.

## Running it

```bash
git clone https://github.com/RachelNefesh/sleep-disorder-ml.git
cd sleep-disorder-ml
pip install -r requirements.txt
jupyter notebook ML_Sleep_Analysis.ipynb
```

Run it top to bottom (**Kernel → Restart & Run All**) to reproduce everything.

## What's in this repo

| File | What it is |
|------|------------|
| `ML_Sleep_Analysis.ipynb` | The full analysis — code, charts, and my write-up of each step |
| `Sleep_health_and_lifestyle_dataset.csv` | The raw dataset |
| `requirements.txt` | Python packages needed to run it |
| `README.md` | This file |
