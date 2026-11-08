# RetailMind – Machine Learning & Deep Learning Project

RetailMind is an end-to-end machine learning and deep learning project focused on customer churn prediction and fashion product image classification.

The project progresses from traditional machine learning models to artificial neural networks (ANNs) and convolutional neural networks (CNNs), covering data preprocessing, feature engineering, model training, evaluation, and model persistence.

---

## Project Objectives

- Predict customer churn using machine learning classification algorithms.
- Perform hyperparameter tuning and feature engineering.
- Compare traditional ML models with an Artificial Neural Network.
- Build a CNN for fashion product image classification.
- Develop reusable data preprocessing and prediction workflows.
- Evaluate models using appropriate classification metrics.

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| Image Processing | PIL, TensorFlow Image API |
| Dataset Download | KaggleHub |
| Model Persistence | Joblib, Keras |
| Development | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |

---

# Project Progress

## Day 1 – Data Collection and Preprocessing

**Objective:** Prepare the Telco Customer Churn dataset for machine learning.

Tasks completed:

- Loaded the Telco Customer Churn dataset.
- Inspected dataset dimensions, data types, and missing values.
- Removed the `customerID` column.
- Converted `TotalCharges` into a numerical column.
- Handled missing values.
- Encoded the target variable `Churn`:
  - No → 0
  - Yes → 1
- Performed stratified train-test splitting.
- Created a reusable preprocessing pipeline using Scikit-learn.

Preprocessing pipeline:

- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Most frequent imputation
  - One-hot encoding

**Dataset:** Telco Customer Churn – 7,043 records.

**Artifacts:**
- `cleaneddata.csv`
- `models/churn_pipeline.joblib`

---

## Day 2 – Model Comparison

**Objective:** Establish machine learning baselines and identify the best-performing model.

Models trained:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**Outcome:** Random Forest was selected as the leading model for further experimentation.

Saved model artifacts:

- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/svm.joblib`

---

## Day 3 – Hyperparameter Tuning

**Objective:** Improve model performance through systematic hyperparameter optimization.

Tasks completed:

- Applied Stratified K-Fold cross-validation.
- Used GridSearchCV for hyperparameter tuning.
- Compared Decision Tree, Random Forest, and SVM.
- Evaluated training and validation recall.
- Selected the tuned Random Forest model as the champion model.

Best cross-validation recall scores:

| Model | Best CV Recall |
|---|---:|
| Random Forest | 0.850167 |
| SVM | 0.835452 |
| Decision Tree | 0.782609 |

**Outcome:** Tuned Random Forest achieved the highest cross-validation recall.

Saved artifacts:

- `models/churn_v3.joblib`
- `reports/tuning_results.json`

---

## Day 4 – Feature Engineering and Ensemble Experiments

**Objective:** Improve churn prediction through domain-based feature engineering and model comparison.

New features engineered:

- `TenureGroup`
- `IsNewCustomer`
- `IsLongTermCustomer`
- `AverageMonthlyCharge`
- `ChargeRatio`
- `ServiceCount`
- `UsesMultipleServices`
- `IsMonthToMonth`
- `HasLongTermContract`
- `UsesElectronicBilling`

Experiments performed:

- Feature importance analysis.
- Feature selection using the fitted Random Forest estimator.
- Training models on engineered features.
- Comparing the original tuned Random Forest with engineered-feature models.
- Experimenting with XGBoost.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

**Outcome:** Compared the impact of feature engineering on model performance.

Saved artifacts (confirm actual files in the repository):

- `models/churn_v4.joblib`
- `reports/day4_model_comparison.csv`

---

## Day 5 – Artificial Neural Network for Customer Churn Prediction

**Objective:** Build a deep learning model to predict customer churn and compare it with the traditional machine learning champion.

### MNIST Warm-up

Before building the churn ANN, explored neural network fundamentals using the MNIST handwritten digit dataset.

Topics covered:

- Neural network architecture
- Input and output layers
- Dense layers
- ReLU activation
- Softmax activation
- Adam optimizer
- Loss functions
- Training and validation accuracy
- Loss visualization across epochs

### ANN for Churn Prediction

Built a feedforward Artificial Neural Network using TensorFlow/Keras.

Architecture:

- Input layer
- Dense layer – 128 neurons, ReLU
- Dropout – 0.3
- Dense layer – 64 neurons, ReLU
- Dropout – 0.2
- Dense layer – 32 neurons, ReLU
- Output layer – 1 neuron, Sigmoid

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Binary Crossentropy
- Batch size: 32
- Maximum epochs: 50
- Early stopping based on validation loss
- Class weighting to address class imbalance

The existing Day 3 preprocessing pipeline was reused to avoid duplicating preprocessing logic.

### Evaluation

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Classification report
- Confusion matrix

Compared the ANN against the tuned Random Forest model from Day 3.

Saved artifacts:

- `models/churn_v5.keras`
- `models/ann_preprocessor.joblib`

---

## Day 6 – Fashion Product Image Classification Using CNN

**Objective:** Build a Convolutional Neural Network to classify real-world fashion product images into their respective categories.

This module introduces computer vision and image classification to the RetailMind project.

### Part 1 – Fashion-MNIST CNN Warm-up

Used the Fashion-MNIST dataset to understand CNN fundamentals.

Dataset:

- 60,000 training images
- 10,000 test images
- Image dimensions: 28 × 28
- Grayscale images
- 10 fashion categories

CNN architecture:

- Conv2D – 32 filters
- MaxPooling2D
- Conv2D – 64 filters
- MaxPooling2D
- Flatten
- Dropout
- Dense – 128 neurons
- Softmax output layer

Topics covered:

- Convolution
- Feature extraction
- Max pooling
- Flattening
- Dropout
- Softmax classification
- Training and validation curves
- Confusion matrix

### Part 2 – Real Fashion Product Dataset

Used the Kaggle dataset:

`paramaggarwal/fashion-product-images-small`

Downloaded using KaggleHub.

Dataset components:

- `styles.csv` – product metadata
- `images/` – fashion product images

The `articleType` column was used as the target classification category.

### Part 3 – Data Preparation

Tasks completed:

- Loaded product metadata.
- Matched product IDs with image filenames.
- Removed records without valid image files or category labels.
- Filtered categories with at least 100 images.
- Encoded category labels using LabelEncoder.
- Performed stratified train-validation-test splitting.

Dataset split:

- 70% Training
- 15% Validation
- 15% Testing

### Part 4 – TensorFlow Image Pipeline

Built a reusable `tf.data` input pipeline.

Image preprocessing:

- Image decoding
- Resizing to 128 × 128
- RGB conversion
- Pixel normalization to the range [0, 1]
- Batching
- Dataset shuffling
- Prefetching

Applied horizontal flipping as a training-time augmentation.

### Part 5 – CNN for Real Product Images

Built a CNN using TensorFlow/Keras.

Architecture:

- Input layer – 128 × 128 × 3
- Conv2D – 32 filters
- MaxPooling2D
- Conv2D – 64 filters
- MaxPooling2D
- Conv2D – 128 filters
- MaxPooling2D
- GlobalAveragePooling2D
- Dense – 128 neurons
- Dropout – 0.4
- Softmax output layer

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Loss: Sparse Categorical Crossentropy
- Metric: Accuracy
- Maximum epochs: 20
- Early stopping based on validation loss
- Model checkpointing to preserve the best model

### Part 6 – Evaluation

Evaluated the trained CNN using:

- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Also analyzed category-level classification performance to identify visually similar categories that may be confused by the model.

### Part 7 – Product Image Prediction

Created a reusable prediction function that:

1. Accepts a product image path.
2. Reads and decodes the image.
3. Resizes it to 128 × 128.
4. Normalizes pixel values.
5. Passes the image to the trained CNN.
6. Returns the predicted product category and confidence score.

Also implemented top-three category predictions.

### Saved Artifacts

- `models/best_product_cnn.keras`
- `models/fashion_product_cnn.keras`
- `models/product_class_mapping.json`
- `models/product_image_config.json`

---

# Repository Structure

The following is the intended project structure. Adjust filenames to match the actual files in the repository.

```text
Retail