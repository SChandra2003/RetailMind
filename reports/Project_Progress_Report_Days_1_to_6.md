# RetailMind Project Progress Report — Detailed Review of Days 1 to 6

**Project:** RetailMind — customer churn prediction and fashion product image classification  
**Progress period in Git history:** 2–10 November 2026  
**Report basis:** README, Python source, notebooks and their saved outputs, JSON/CSV reports, model/configuration artifacts, tests, and Git commit history.

## 1. Executive summary

Over the first six project days, I developed two related machine learning tracks:

1. A customer churn workflow that moves from data inspection and a Logistic Regression baseline to model comparison, cross-validation tuning, and feature engineering.
2. An image classification workflow that uses Fashion-MNIST to practice CNNs and then applies a CNN to downloaded fashion product images.

The repository contains the churn source data, preprocessing and prediction code, saved churn models, model comparison results, and saved product image CNN artifacts. The first four days have notebooks with saved results that support the reported churn metrics. Day 5 is described in the README and has model files, but its notebook is empty. The Day 6 notebook contains the model workflow, but its current version does not preserve evaluation outputs. These differences matter when reproducing or presenting the project results.

## 2. Day-by-day work performed

### Day 1 — Data ingestion, exploration, preparation, and baseline

**Goal:** Prepare the Telco Customer Churn data for an initial predictive model.

**Work performed**

- Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`, containing 7,043 customer records and 21 columns.
- Inspected the sample records, dimensions, data types, target distribution, numeric summaries, duplicate rows, and duplicate customer IDs. The notebook recorded no duplicate rows or duplicate customer IDs.
- Identified `customerID` as an identifier rather than a predictive feature and removed it from the feature matrix.
- Mapped the target values `No` and `Yes` to 0 and 1.
- Found that `TotalCharges` was read as text. The exploration notebook found 11 blank strings; the training notebook converted blanks and invalid values to missing numeric values.
- Built preprocessing with a `ColumnTransformer`:
  - Numeric columns: median imputation followed by standard scaling.
  - Categorical columns: most-frequent imputation followed by one-hot encoding.
  - `OneHotEncoder(handle_unknown="ignore")` allows prediction data to contain a category absent during fitting.
- Used a stratified 80/20 split with `random_state=42`: 5,634 training rows and 1,409 test rows.
- Trained a Logistic Regression pipeline with `max_iter=1000` and `class_weight="balanced"`.
- Saved the model as `models/churn_pipeline.joblib` and held-out features plus labels as `data/processed/test_data.csv`.

**Data profile**

| Finding | Value |
|---|---:|
| Customers | 7,043 |
| Churn: No | 5,174 (73.46%) |
| Churn: Yes | 1,869 (26.54%) |
| Blank `TotalCharges` strings | 11 |
| Duplicate rows / duplicate customer IDs | 0 / 0 |
| Median / average tenure | 29 / 32.37 months |

**Baseline test results**

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |

The confusion matrix in the evaluation notebook is `[[747, 288], [81, 293]]` using the order No Churn, Churn. This means the baseline correctly identified 293 of 374 churners in the held-out data, while 81 churners were missed. It also flagged 288 customers who did not churn.

**Artifacts:** `models/churn_pipeline.joblib`, `data/processed/test_data.csv`, and `reports/baseline_metrics.json`.

### Day 2 — Comparison of traditional classifiers

**Goal:** Compare the baseline with tree and margin-based classifiers on the same split.

**Work performed**

- Reused the Day 1 preprocessing pipeline and the same stratified 80/20 split.
- Trained Decision Tree, Random Forest, and RBF-kernel SVM models. The tree models and SVM used class weighting; the Random Forest used 300 trees, and the SVM enabled probability estimates for ROC-AUC evaluation.
- Compared all four models (including the Logistic Regression baseline) using accuracy, precision, recall, F1, and ROC-AUC.
- Saved the three additional estimators and wrote results to `reports/model_comparison.json`.

**Test-set comparison**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7381 | 0.5043 | **0.7834** | 0.6136 | **0.8413** |
| Decision Tree | 0.7317 | 0.4946 | 0.4866 | 0.4906 | 0.6537 |
| Random Forest | **0.7658** | **0.5521** | 0.6230 | 0.5854 | 0.8218 |
| SVM | 0.7438 | 0.5115 | 0.7727 | **0.6155** | 0.8211 |

Random Forest led on accuracy and precision. Logistic Regression led on recall and ROC-AUC, while SVM had the highest F1 by a small margin. The README names Random Forest as the leading model for further experimentation; the comparison results show that this choice is strongest if accuracy is the main criterion. For churn detection, recall is also important because a false negative is a customer who churns but is not identified.

**Artifacts:** `models/decision_tree.joblib`, `models/random_forest.joblib`, `models/svm.joblib`, and `reports/model_comparison.json`.

### Day 3 — Cross-validation and hyperparameter tuning

**Goal:** Search for model settings that improve churn recall and assess the train/validation gap.

**Work performed**

- Used five-fold `StratifiedKFold` cross-validation with shuffling and a fixed random seed.
- Tuned the Decision Tree, Random Forest, and SVM using `GridSearchCV` with recall as the scoring metric and `n_jobs=-1` to run folds in parallel.
- Evaluated 148 parameter combinations, or 740 model fits across five folds: 36 Decision Tree combinations, 96 Random Forest combinations, and 16 SVM combinations.
- Compared training and validation recall, reviewed a learning curve, and evaluated the selected Random Forest on the held-out test data.
- Saved the selected estimator to `models/churn_v3.joblib` and the cross-validation results to `reports/tuning_results.json`.

**Best cross-validation results and parameters**

| Model | Best CV recall | Best parameters |
|---|---:|---|
| Random Forest | **0.8502** | `n_estimators=200`, `max_depth=5`, `max_features=sqrt`, `min_samples_leaf=5`, `min_samples_split=2` |
| SVM | 0.8355 | `C=100`, `gamma=auto`, `kernel=rbf` |
| Decision Tree | 0.7826 | `max_depth=5`, `min_samples_leaf=5`, `min_samples_split=2` |

**Tuned Random Forest holdout results:** accuracy 0.6856, precision 0.4504, recall 0.8369, F1 0.5856, ROC-AUC 0.8106. This model identified more churners than the baseline (recall 0.8369 versus 0.7834), at the cost of lower accuracy and precision. That trade-off follows from optimizing the search for recall.

**Artifacts:** `models/churn_v3.joblib` and `reports/tuning_results.json`.

### Day 4 — Domain-based feature engineering and ensemble experiments

**Goal:** Test whether additional customer-level information improves the churn models.

**Work performed**

- Started with the same 19 input features used in the earlier churn work and created 10 additional features, producing 29 columns before model preprocessing.
- Added tenure group flags, charge-derived values, service counts, contract flags, and an electronic billing flag.
- Reviewed the Day 3 Random Forest feature importances. Leading signals included contract term, lack of streaming/internet services, online security, tenure, and online backup.
- Built an engineered-feature Random Forest and an XGBoost model and compared both against the Day 3 Random Forest using the same held-out set.
- Saved the comparison CSV and saved the notebook-selected champion as `models/churn_v4.joblib`.

**Engineered features**

| Feature | Meaning in the notebook |
|---|---|
| `TenureGroup` | New, short-term, medium-term, long-term, or loyal tenure band |
| `IsNewCustomer` | Tenure at or below 12 months |
| `IsLongTermCustomer` | Tenure at or above 48 months |
| `AverageMonthlyCharge` | `TotalCharges / tenure` |
| `ChargeRatio` | `TotalCharges / (MonthlyCharges × tenure)` |
| `ServiceCount` | Number of listed services marked Yes |
| `UsesMultipleServices` | At least three services marked Yes |
| `IsMonthToMonth` | Month-to-month contract flag |
| `HasLongTermContract` | One-year or two-year contract flag |
| `UsesElectronicBilling` | Paperless billing flag |

Zero tenure was replaced with a missing denominator for the charge features. The notebook observed eight missing values in each charge-derived feature and the new preprocessing pipeline median-imputed numeric features.

**Holdout comparison**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Day 3 Random Forest | 0.6856 | 0.4504 | **0.8369** | 0.5856 | 0.8106 |
| Day 4 Random Forest | 0.7644 | 0.5379 | 0.7968 | **0.6422** | **0.8417** |
| Day 4 XGBoost | **0.7963** | **0.6495** | 0.5053 | 0.5684 | 0.8407 |

XGBoost led on accuracy and precision, while the engineered Random Forest led on F1 and ROC-AUC. The Day 3 Random Forest retained the highest recall and was selected as champion by the notebook's ranking. The notebook then saved that selected champion under the Day 4 artifact name `churn_v4.joblib`.

**Feature-selection note:** the `SelectFromModel` output reports 5,320 selected features out of 5,320 transformed features. The selection step ran, but its saved output shows no reduction in the transformed feature count.

**Artifacts:** `models/churn_v4.joblib` and `reports/day4_model_comparison.csv`.

### Day 5 — Artificial Neural Network for churn prediction

**Goal as documented:** Learn the ANN workflow and compare a feedforward neural network against the traditional churn champion.

The README describes a MNIST warm-up and a TensorFlow/Keras feedforward network with the following structure:

`Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, Sigmoid)`

The documented training setup is Adam with learning rate 0.001, binary cross-entropy, batch size 32, up to 50 epochs, early stopping on validation loss, and class weighting for churn imbalance. The README says the Day 3 preprocessing approach was reused and that evaluation was planned with accuracy, precision, recall, F1, ROC-AUC, a classification report, and a confusion matrix.

The repository contains `models/churn_v5.keras` and `models/ann_preprocessor.joblib`. These files support that model artifacts were saved, but they do not provide the missing training notebook or measured results.

**Evidence limitation:** `notebooks/09_ann_churn_model.ipynb` is zero bytes. The README also says the final Day 5 evaluation metrics still need to be added. The architecture and settings above are therefore reported as README documentation, not independently confirmed against runnable notebook code.

### Day 6 — Fashion-MNIST and fashion product image CNNs

**Goal:** Practice convolutional image classification and then train a model on real product images.

#### Fashion-MNIST warm-up

- Used 28×28 grayscale images with 10 clothing classes (60,000 training and 10,000 test images, as documented in the README).
- Scaled pixel values by 255 and added a channel dimension.
- Built a CNN with Conv2D layers using 32 and 64 filters, max pooling, flattening, dropout at 0.3, a Dense(128, ReLU) layer, and a 10-unit softmax output.
- Compiled with Adam and sparse categorical cross-entropy; trained with batch size 64, validation split 0.2, and up to 15 epochs. Early stopping watched validation loss with patience 3 and restored the best weights.
- Added evaluation, training/validation curves, a confusion matrix, and a classification report.

The current notebook source contains these steps, but the latest notebook file does not preserve output metrics, so no Fashion-MNIST test score is included in this report.

#### Real fashion product images

- Used KaggleHub to download `paramaggarwal/fashion-product-images-small` and searched the downloaded directory recursively for `styles.csv`.
- Loaded metadata with malformed CSV rows skipped. Matched `id` values to `<id>.jpg` filenames and removed rows without `articleType` values or image files.
- Counted `articleType` values and retained rows whose `articleType` had at least 100 examples.
- Encoded `masterCategory` as the model target, creating five classes: Accessories, Apparel, Footwear, Free Items, and Personal Care.
- Split records into 70% training, 15% validation, and 15% test data with stratification on the encoded `masterCategory` label.
- Created a `tf.data` pipeline that reads JPEG files, decodes RGB, resizes to 128×128, converts pixel values to float values in [0, 1], batches at 32, shuffles training records, applies random horizontal flipping during training, and prefetches batches.
- Built a CNN with three Conv2D/MaxPooling blocks (32, 64, and 128 filters), Global Average Pooling, Dense(128, ReLU), Dropout(0.4), and a softmax output. It uses Adam at 0.001 and sparse categorical cross-entropy.
- Trained for up to 15 epochs with early stopping on validation loss (patience 4, restore best weights) and model checkpointing to `models/best_product_cnn.keras`.
- Added weighted accuracy/precision/recall/F1 reporting, a classification report and confusion matrix, a prediction function for one image, and top-three category probabilities.
- Saved and reloaded the model and wrote a label mapping and image configuration for use with predictions.

**Target-label clarification:** the notebook filters examples using `articleType`, but the label encoder trains on `masterCategory`. The saved mapping confirms the five broad `masterCategory` labels. Thus the current classifier is a five-class broad-category classifier; it is not an `articleType` classifier as the README currently states. The README should be aligned with the implementation, or the notebook should be changed consistently if `articleType` was intended.

**Artifacts:** `models/best_product_cnn.keras`, `models/fashion_product_cnn.keras`, `models/product_class_mapping.json`, and `models/product_image_config.json` (128×128 RGB, normalized by dividing pixel values by 255, five classes).

**Evidence limitation:** the current notebook has evaluation code but no saved evaluation outputs. The README explicitly says final Day 6 metrics need to be recorded. The Kaggle dataset itself is downloaded separately and is not included in the repository's `data/` directory.

## 3. Issues, mitigations, and remaining risks

| Issue or risk | Handling visible in the project | Status |
|---|---|---|
| `TotalCharges` arrives as text and has 11 whitespace-only values. | Convert with `pd.to_numeric(..., errors="coerce")`; median-impute within the numeric pipeline. | Addressed in preprocessing code. |
| Churn is imbalanced (26.54% churn). | Stratify train/test and CV splits; apply `class_weight="balanced"` in the baseline and several candidate models. | Addressed for those estimators. Day 4 XGBoost does not use class weighting in its current configuration. |
| Unseen categories at inference time can break one-hot encoding. | Set `handle_unknown="ignore"`; include a test case that fits and predicts with a previously unseen contract category. | Addressed in code; this report did not execute the tests. |
| Charge-derived features can divide by zero or become missing. | Replace zero denominators with NaN and run the engineered features through an imputer. | Addressed in Day 4 notebook; eight missing values per derived charge feature were observed. |
| Models rank differently depending on the selected metric. | Report multiple metrics and select the tuned/Day 4 champion by recall. | Addressed as a model-selection choice; it trades away some accuracy/precision for fewer missed churners. |
| Grid search requires many training runs. | Use five-fold stratified CV and `n_jobs=-1`; 740 total fits are reported. | Search is automated and parallelized; no elapsed-time measurement is saved. |
| Downloaded product metadata can contain malformed rows, and some image paths or labels may be absent. | Skip malformed CSV lines, locate `styles.csv`, verify image existence, remove missing labels, and filter low-count article types. | Handling is in the notebook; the external dataset must still be downloaded and accessible. |
| Notebook working directory can vary. | Some notebooks move to the parent project directory or search for data files. | Partially addressed in notebook path setup; running order and working directory still matter. |
| Day 6 target description and implementation disagree. | The report uses the notebook and saved class mapping as evidence: `masterCategory` is the target. | Unresolved in project documentation. |
| Day 5 implementation cannot be reproduced from its notebook. | The README and saved model files document the experiment, but the notebook is empty. | Unresolved; restore or recreate `09_ann_churn_model.ipynb`. |
| Day 5 and Day 6 evaluation numbers are absent from the checked-in record. | Evaluation code exists for Day 6; the README calls out missing final metrics. | Unresolved; run the notebooks in the intended environment and save final outputs. |
| Notebook dependencies are not fully listed. | `requirements.txt` currently lists pandas, scikit-learn, and joblib. The notebooks also use TensorFlow/Keras, matplotlib, seaborn, XGBoost, KaggleHub, and PIL. | Unresolved setup/reproducibility gap; document the full environment before rerunning all days. |
| scikit-learn emitted a warning for `SVC(probability=True)` in the saved outputs. | No replacement or compatibility change is recorded in the repository. | Not shown as resolved; future scikit-learn versions may require probability calibration instead. |

## 4. Deliverables present in the repository

- **Data:** raw Telco churn CSV, processed `cleaned_data.csv`, and held-out `test_data.csv`.
- **Reusable churn code:** `src/retailmind/pipeline.py` builds preprocessing/classifier pipelines; `src/retailmind/service.py` loads a saved pipeline and returns churn probabilities and labels.
- **Churn models:** baseline, Day 2 model comparison artifacts, tuned Day 3 model, Day 4 selected champion, and Day 5 ANN artifacts.
- **Reports:** baseline metrics, model comparison JSON, tuning results JSON, and Day 4 model comparison CSV.
- **Image classification:** Day 6 notebook, CNN models, class mapping, and preprocessing configuration.
- **Basic pipeline coverage:** `tests/test_pipeline.py` checks that the pipeline can fit and predict with an unseen categorical value. No tests were run while preparing this report.

## 5. Recommended follow-up to make the project presentation complete

1. Restore or recreate the Day 5 ANN notebook so the documented network can be checked and rerun.
2. Record final Day 5 ANN and Day 6 image-model metrics, confusion matrices, and class reports in saved notebook outputs or report files.
3. Decide whether Day 6 predicts `masterCategory` or `articleType`, then align the label construction, category filtering, README, class mapping, and evaluation labels.
4. Expand `requirements.txt` or provide an environment file covering the libraries used across all notebooks, especially TensorFlow/Keras, XGBoost, KaggleHub, plotting libraries, and PIL.
5. Confirm the Day 4 feature-selection intent: the saved output says all 5,320 transformed features were selected.
6. Review the SVM probability warning against the installed scikit-learn version and update probability estimation if required.
7. Correct the README's `cleaneddata.csv` reference to the actual filename `data/processed/cleaned_data.csv`.

## 6. Overall assessment

By the end of Day 6, the project has a substantial end-to-end structure: data inspection, reusable preprocessing, model comparison, cross-validated tuning, feature engineering, serialized models, and an image classification pipeline with single-image inference. The churn results are supported by saved notebook outputs and report files through Day 4. Day 5 and Day 6 demonstrate saved model artifacts and documented workflows, but their final performance results and reproducibility are incomplete in the current checkout. The Day 6 target-label mismatch should be resolved before describing that model's task or interpreting its class-level results.
