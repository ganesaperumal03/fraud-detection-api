# fraud-detection-api

A fraud detection model built on the PaySim dataset and served with FastAPI.

The model looks at one transaction and gives a score between 0 and 1.
A higher score means the transaction is more likely to be fraud.

Status: model, threshold and API are done. Logging, Docker, monitoring and
the rules-vs-ML comparison are still to be added.

---

## 1. The dataset

### What it is

PaySim is simulated mobile money data, from Kaggle
("Synthetic Financial Datasets For Fraud Detection"). It copies how a real
mobile wallet behaves over 30 days.

- 6,362,620 transactions
- 8,213 are fraud (0.13%, about 1 in 775)

### Why I chose it

- It looks like fintech data: transfers, cash-outs and payments in a wallet app.
- It has labels, so we know which transactions were fraud.
- It is large and very imbalanced, like real fraud data.
- The columns have clear names, so every feature can be explained.
- It is simulated, so there is no real customer data and it is safe to share.

### The columns

| Column | Meaning |
|---|---|
| `step` | Time. 1 step = 1 hour. The data covers 743 hours (30 days) |
| `type` | Kind of transaction: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Money moved |
| `nameOrig` | Sender's ID |
| `oldbalanceOrg` | Sender's balance before the transaction |
| `newbalanceOrig` | Sender's balance after the transaction |
| `nameDest` | Receiver's ID |
| `oldbalanceDest` | Receiver's balance before the transaction |
| `newbalanceDest` | Receiver's balance after the transaction |
| `isFraud` | The label: 1 = fraud, 0 = normal. This is what the model predicts |
| `isFlaggedFraud` | An old simple rule that flags very big transfers. Not used |

---

## 2. Step by step: what I did and what I found

### Step 1: Look at the data

| What I checked | What I found |
|---|---|
| Size | 6,362,620 rows |
| Fraud rows | 8,213 (0.13%) |
| Fraud by type | Only in CASH_OUT (4,116) and TRANSFER (4,097). None in the other three types |
| Senders | 6,353,307 different senders. Almost every sender appears only once |
| Receivers | 2,722,362 different receivers. Receivers repeat |

What this told me:

- The data is very imbalanced, so accuracy cannot be used (see section 3).
- `type` is a useful feature.
- Sender history is useless here, because senders do not repeat. Receiver
  history can be used.

### Step 2: Baseline model (logistic regression)

A baseline is a simple first model. It gives a score to beat.

I started with logistic regression because the answer is yes/no (fraud or
not), it trains fast, and it is easy to explain.

Features used: `amount`, `oldbalanceOrg`, `oldbalanceDest`, `hour`, and
`type` as one-hot columns.

Result: PR-AUC 0.154, precision 0.019, recall 0.940.

What I found: it catches most fraud, but only about 2 of every 100 alerts
are real fraud. That is about 80,000 alerts to find about 1,540 frauds.
Not usable.

Why it is weak: logistic regression learns simple patterns like "bigger
amount means more risk". It cannot easily learn mixed patterns like
"transfer AND amount equal to the full balance".

### Step 3: XGBoost with the same features

Same features, same split. Only the model changed.

Result: PR-AUC 0.909, precision 0.201, recall 0.991.

What I found: alerts fell from about 80,000 to about 8,100 and it caught
more fraud. A tree model can mix several features together, which
logistic regression could not do.

### Step 4: Add new features

Same model and settings. Only the features changed. (Features are listed
in section 5.)

Result: PR-AUC 0.989, precision 0.874, recall 0.995.

What I found: false alarms fell from about 6,450 to about 235. Good
features helped as much as a better model.

### Step 5: Pick the threshold

I tried many thresholds and put a cost on each. Details in section 6.

Result: threshold 0.30.

### Step 6: Final serving model

The API sees one transaction at a time and does not know the receiver's
history. So the final model leaves out the three velocity features.

Result: PR-AUC 0.989. The same as before, so nothing was lost.

---

## 3. How the model is measured

### Why not accuracy

Accuracy = correct answers / total answers.

A model that says "not fraud" for every transaction would be 99.87%
accurate on this data and would catch zero fraud. So accuracy is
misleading when fraud is rare.

### Precision and recall

| Measure | Question it answers | Low value means |
|---|---|---|
| Recall | Of all the real fraud, how much did we catch? | Fraud gets through. The bank loses money |
| Precision | Of everything we flagged, how much was really fraud? | Many false alarms. Good customers get blocked |

Usually when one goes up, the other goes down.

### PR curve and PR-AUC

The model gives a score, and a threshold turns it into a flag. Each
threshold gives a different precision and recall. The PR curve draws
precision and recall for every threshold as one line.

PR-AUC is the area under that line, as one number from 0 to 1. Higher is
better. A model that only guesses gets about the fraud rate (0.0013 here).

I use PR-AUC to compare models because it does not depend on one threshold.

### ROC curve, and why I did not rely on it

The ROC curve also tries every threshold, but it plots recall against the
false positive rate (of all normal transactions, how many were flagged by
mistake).

With millions of normal transactions, the false positive rate looks tiny
even when most alerts are wrong. So ROC can look good on fraud data when
the model is not. The PR curve does not hide this. That is why PR-AUC is
the main number in this project.

---

## 4. Why these models

### Why logistic regression first

It is the simple baseline. Without it, there is no way to know if a
complex model is really better.

### Why XGBoost next

- It usually gives the best results on table data (rows and columns).
- It can learn mixed patterns across several features.
- It trains fast on millions of rows (`tree_method="hist"`).
- `scale_pos_weight` handles the imbalance.
- It is widely used in fraud teams.

### Why not the others

| Model | Why not here |
|---|---|
| Linear regression | It predicts a number, not a yes/no answer |
| Single decision tree | Easy to explain, but weak alone |
| Random forest | Would work. XGBoost usually scores a little higher on table data and is faster on this much data |
| SVM | Too slow on 6 million rows |
| KNN | Too slow on 6 million rows |
| Deep learning | Made for text, images and audio. Not needed for a small table of numbers |

### How the imbalance was handled

Class weights. The model is told that missing a fraud is a much bigger
mistake than getting a normal transaction wrong.

- Logistic regression: `class_weight="balanced"`
- XGBoost: `scale_pos_weight` = normal rows / fraud rows = about 774

The test data was not changed, so precision and recall are honest.

---

## 5. Features

| Feature | Meaning | Kind |
|---|---|---|
| `hour` | Hour of the day, from `step % 24` | Time of day |
| `amount_to_balance` | Amount divided by the sender's balance. Close to 1 means "sending almost everything" | Amount pattern |
| `empties_account` | 1 if the payment takes the full balance or more | Amount pattern |
| `dest_count_before` | Payments this receiver got before this one | Velocity |
| `dest_amount_before` | Money this receiver got before this one | Velocity |
| `dest_count_this_hour` | Payments this receiver already got in the same hour | Velocity |
| `type_...` | Transaction type as one-hot columns | Type |

### No data leakage

Every feature uses only information from before the payment.

- The "new balance" columns are not used, because they are known only
  after the payment.
- Velocity features count only earlier rows for that receiver.

### Which features mattered

| Feature | Importance |
|---|---|
| `empties_account` | 0.764 |
| `amount_to_balance` | 0.209 |
| `amount` | 0.009 |
| All others together | about 0.02 |

The model mostly learned one pattern: fraud empties the account.

The velocity features added very little on this dataset. They also need
the receiver's history, which the API cannot know from one request
without a feature store. So the serving model leaves them out.

---

## 6. How the results improved

Same 80/20 stratified split for all models. Precision and recall are at
threshold 0.5.

| Step | What changed | PR-AUC | Precision | Recall |
|---|---|---|---|---|
| Logistic regression | Baseline | 0.154 | 0.019 | 0.940 |
| XGBoost, simple features | Better model | 0.909 | 0.201 | 0.991 |
| XGBoost, new features | Better features | 0.989 | 0.874 | 0.995 |
| XGBoost, serving model | Velocity features removed | 0.989 | - | - |

Fill in the last row's precision and recall from your own run.

In counts, on about 1,640 frauds in the test data:

| Model | Frauds caught | Alerts | False alarms |
|---|---|---|---|
| Logistic regression | about 1,540 | about 80,000 | about 78,500 |
| XGBoost, simple features | about 1,630 | about 8,100 | about 6,450 |
| XGBoost, new features | about 1,635 | about 1,870 | about 235 |

---

## 7. Threshold and business trade-off

The threshold is the cut-off line. A score above it is flagged.

- Lower threshold: catches more fraud, but more false alarms.
- Higher threshold: fewer false alarms, but more fraud is missed.

The default 0.5 has no business meaning. So I chose the threshold by cost.

Assumed costs: a missed fraud costs Rs 20,000. A false alarm costs
Rs 1,000 (review time and an unhappy customer).

Cost = missed frauds x 20,000 + false alarms x 1,000

| Threshold | Alerts | Caught | Missed | False alarms | Precision | Recall | Cost (Rs) |
|---|---|---|---|---|---|---|---|
| 0.10 | 1947 | 1639 | 4 | 308 | 0.842 | 0.998 | 388,000 |
| 0.20 | 1914 | 1638 | 5 | 276 | 0.856 | 0.997 | 376,000 |
| **0.30** | **1893** | **1638** | **5** | **255** | **0.865** | **0.997** | **355,000** |
| 0.40 | 1882 | 1637 | 6 | 245 | 0.870 | 0.996 | 365,000 |
| 0.50 | 1871 | 1635 | 8 | 236 | 0.874 | 0.995 | 396,000 |
| 0.60 | 1864 | 1635 | 8 | 229 | 0.877 | 0.995 | 389,000 |
| 0.70 | 1855 | 1634 | 9 | 221 | 0.881 | 0.995 | 401,000 |
| 0.80 | 1842 | 1631 | 12 | 211 | 0.885 | 0.993 | 451,000 |
| 0.90 | 1827 | 1626 | 17 | 201 | 0.890 | 0.990 | 541,000 |
| 0.95 | 1816 | 1619 | 24 | 197 | 0.892 | 0.985 | 677,000 |
| 0.99 | 1744 | 1587 | 56 | 157 | 0.910 | 0.966 | 1,277,000 |

Chosen threshold: **0.30**, the lowest total cost. It catches 99.7% of
fraud, and 86.5% of alerts are real fraud. That is about 315 alerts a day
across the whole system, which a small review team can handle.

### The threshold depends on the business cost

| False alarm cost | Best threshold |
|---|---|
| Rs 100 | 0.10 |
| Rs 1,000 | 0.30 |

The model is the same in both rows. When false alarms cost more, the best
threshold moves up.

---

## 8. Honest limits

- PaySim is simulated. Its fraud follows a simple pattern (the fraudster
  empties the account), so scores here are higher than on real bank data.
- The cost numbers are assumptions. A real bank would give its own.
- The threshold was chosen on the test set. In production it should be
  chosen on a separate validation set.
- The threshold table comes from the model with velocity features. It
  should be re-run on the final serving model to confirm 0.30 is still best.
- The split is random. A split by time (train on early days, test on later
  days) would be closer to real use.

---

## 9. What is in this repo

| File | What it is |
|---|---|
| `notebooks/fraud_model.ipynb` | All steps: data, baseline, XGBoost, features, threshold, save the model |
| `app.py` | FastAPI service with input checks |
| `model/` | The saved model goes here (`fraud_model.joblib`) |
| `data/` | Put `paysim.csv` here (not pushed to GitHub) |
| `requirements.txt` | The Python packages needed |

## 10. How to run

1. Install the packages:

   ```
   pip install -r requirements.txt
   ```

2. Download the PaySim CSV from Kaggle and save it as `data/paysim.csv`.

3. Open `notebooks/fraud_model.ipynb` and run all cells. This trains the
   models and saves `model/fraud_model.joblib`.

4. Start the API from the repo folder:

   ```
   uvicorn app:app --reload
   ```

5. Open http://127.0.0.1:8000/docs and try `POST /predict` with:

   ```json
   {"step": 1, "type": "TRANSFER", "amount": 181, "oldbalanceOrg": 181, "oldbalanceDest": 0}
   ```

### Input checks

The API rejects a request before it reaches the model if:

- `step` is less than 1
- `type` is not one of CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- `amount` is 0 or negative
- a balance is negative
- any field is missing