# Customer Support Annotation & Intent Classification

A machine learning project for classifying customer support messages into banking-related intents using the Banking77 dataset.

## Live Demo

**Try the application:** https://customer-support-ai-gkmuxmcdqfjcwk3q6tpidm.streamlit.app/

## Project Overview

This project demonstrates an end-to-end customer support text classification workflow:

```text
Raw Dataset
    ↓
Dataset Inspection
    ↓
Data Cleaning
    ↓
Train / Validation / Test Split
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Error Analysis
    ↓
Interactive Prediction
```

The system takes a customer message as input and predicts the most likely customer-support intent.

## Example

```text
Customer message:
My card hasn't arrived yet

Predicted Intent:
card_arrival
```

Another example:

```text
Customer message:
I want to cancel my transfer

Predicted Intent:
cancel_transfer
```

## Dataset

The project uses the Banking77 dataset, which contains customer banking queries categorized into 77 different intents.

Examples of intents include:

* `card_arrival`
* `card_not_working`
* `cancel_transfer`
* `declined_transfer`
* `declined_cash_withdrawal`
* `pending_card_payment`
* `cash_withdrawal_not_recognised`
* `verify_my_identity`
* `top_up_failed`
* `transfer_not_received_by_recipient`

The raw dataset is stored in:

```text
data/raw/train.csv
```

## Project Structure

```text
customer-support-annotation/
│
├── annotation/
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── models/
│   └── customer_support_model.pkl
│
├── reports/
│   └── validation_predictions.csv
│
├── scripts/
│   ├── clean_dataset.py
│   ├── error_analysis.py
│   ├── evaluate_model.py
│   ├── inspect_dataset.py
│   ├── predict.py
│   ├── split_dataset.py
│   ├── test_model.py
│   └── train_model.py
│
└── README.md
```

## Data Processing

The dataset was processed through several stages:

### 1. Dataset Inspection

`inspect_dataset.py` was used to inspect:

* Dataset structure
* Number of records
* Column names
* Missing values
* Unique categories
* Duplicate records
* Sample records

### 2. Data Cleaning

`clean_dataset.py` prepares the raw data for machine learning by handling data-quality issues before training.

### 3. Dataset Splitting

`split_dataset.py` separates the dataset into training, validation, and test sets.

The evaluation workflow uses separate validation and test datasets so that model performance can be measured on unseen data.

## Model Training

The model is trained using:

```text
scripts/train_model.py
```

The trained model is saved as:

```text
models/customer_support_model.pkl
```

## Model Evaluation

### Validation Results

Validation set:

```text
Records: 1,000
Accuracy: 83.90%
Macro F1: 0.83
Weighted F1: 0.84
```

### Final Test Results

The final model was evaluated on 1,000 unseen test records.

```text
Test records: 1,000
Accuracy: 84.70%
Macro F1: 0.85
Weighted F1: 0.85
```

### Final Performance

| Metric      | Validation |       Test |
| ----------- | ---------: | ---------: |
| Records     |      1,000 |      1,000 |
| Accuracy    |     83.90% | **84.70%** |
| Macro F1    |       0.83 |   **0.85** |
| Weighted F1 |       0.84 |   **0.85** |

The final test accuracy of **84.70%** indicates that the model correctly classified approximately 847 out of 1,000 unseen customer messages.

## Error Analysis

The project includes an error-analysis workflow using:

```text
scripts/error_analysis.py
```

During validation:

```text
Validation records: 1,000
Correct predictions: 839
Incorrect predictions: 161
Error rate: 16.10%
```

The analysis showed that the model mainly struggles with semantically similar intents.

For example:

```text
Actual:
why_verify_identity

Predicted:
verify_my_identity
```

and:

```text
Actual:
topping_up_by_card

Predicted:
pending_top_up
```

These errors demonstrate the difficulty of distinguishing between closely related customer-support intents.

The validation prediction results are saved to:

```text
reports/validation_predictions.csv
```

## Interactive Prediction

The project includes an interactive prediction script:

```text
scripts/predict.py
```

Run it with:

```bash
python scripts/predict.py
```

Example:

```text
===== CUSTOMER SUPPORT AI =====
Type 'exit' to stop.

Customer message: My card hasn't arrived yet
Predicted Intent: card_arrival
```

The program continues accepting customer messages until:

```text
exit
```

is entered.

## Testing

The final test evaluation can be run with:

```bash
python scripts/test_model.py
```

This evaluates the trained model on the held-out test dataset.

## Main Scripts

| Script               | Purpose                          |
| -------------------- | -------------------------------- |
| `inspect_dataset.py` | Inspects the raw dataset         |
| `clean_dataset.py`   | Cleans and prepares the data     |
| `split_dataset.py`   | Creates dataset splits           |
| `train_model.py`     | Trains the classification model  |
| `evaluate_model.py`  | Evaluates validation performance |
| `test_model.py`      | Performs final test evaluation   |
| `error_analysis.py`  | Analyzes incorrect predictions   |
| `predict.py`         | Provides interactive predictions |

## Limitations

This is a closed-set intent classification system. The model predicts one of the known Banking77 intents for an input message.

For example, an unrelated message such as:

```text
Weather is not good today
```

may still be assigned to one of the available banking intents because the current classifier does not have a dedicated `unknown` or `out_of_domain` class.

This is an important limitation when using the model with real-world customer messages.

## Key Learning Outcomes

This project provided practical experience with:

* Dataset inspection
* Data cleaning
* Text classification
* Label/intent handling
* Train/validation/test splitting
* Machine learning model training
* Model evaluation
* Precision, recall and F1-score
* Error analysis
* Model serialization with Joblib
* Interactive model inference
* Organizing an end-to-end ML project

## Final Result

The completed customer support classification system achieved:

**84.70% accuracy on 1,000 unseen test records.**

The project demonstrates a complete machine-learning workflow from raw customer-support data to a trained and evaluated intent-classification model.
