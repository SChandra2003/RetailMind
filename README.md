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
RetailMind/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 06_model_comparison.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_feature_engineering_ensembles.ipynb
│   ├── 09_ann_churn_model.ipynb
│   └── 10_product_image_classifier.ipynb
│
├── models/
│   ├── churn_pipeline.joblib
│   ├── churn_v3.joblib
│   ├── churn_v4.joblib
│   ├── churn_v5.keras
│   ├── ann_preprocessor.joblib
│   ├── best_product_cnn.keras
│   ├── fashion_product_cnn.keras
│   ├── product_class_mapping.json
│   └── product_image_config.json
│
├── reports/
│   ├── tuning_results.json
│   └── day4_model_comparison.csv
│
├── src/
│   └── retailmind/
│       └── pipeline.py
│
├── tests/
│
├── README.md
└── requirements.txt
```

---

# Key Learning Outcomes

By completing the first six days, the project covers:

- Data cleaning and preprocessing
- Handling missing values
- Encoding categorical variables
- Feature engineering
- Scikit-learn pipelines
- Classification algorithms
- Cross-validation
- Hyperparameter tuning
- Feature selection
- Artificial Neural Networks
- Convolutional Neural Networks
- Image preprocessing
- TensorFlow data pipelines
- Data augmentation
- Model evaluation
- Confusion matrix analysis
- Model serialization and reuse
- Individual image inference

---

# How to Run the Project

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies.
4. Launch Jupyter Notebook.
5. Run the notebooks in sequence, using the appropriate dataset and model artifacts.

Example:

```bash
git clone https://github.com/SChandra2003/RetailMind.git

cd RetailMind

pip install -r requirements.txt

jupyter notebook
```

Note: The real fashion product dataset is downloaded separately through KaggleHub. Ensure Kaggle access is configured before running the Day 6 notebook.

---

# Future Improvements

Potential next steps:

- Transfer learning using MobileNetV2 or EfficientNet.
- Compare CNN performance with pretrained models.
- Improve generalization through data augmentation.
- Build a product image classification interface.
- Deploy the trained models through a FastAPI application.
- Integrate churn prediction and product classification into a unified application.

---

## Project Status

| Day | Module | Status |
|---|---|---|
| Day 1 | Data preprocessing | Completed |
| Day 2 | Model comparison | Completed |
| Day 3 | Hyperparameter tuning | Completed |
| Day 4 | Feature engineering and ensembles | Completed |
| Day 5 | ANN churn prediction | Completed |
| Day 6 | Fashion product CNN classification | Completed |

## Day 9 – RNN / LSTM Based Demand Forecasting

### Objective
Implemented a time-series demand forecasting module using an LSTM (Long Short-Term Memory) neural network to predict the next day's store sales based on historical sales data. The objective is to support better inventory planning and reduce the risk of overstocking and stockouts.

### Dataset
- **Dataset:** Rossmann Store Sales
- **Source:** [Kaggle – Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales/data)
- Selected Store 1 for the initial forecasting implementation.
- Used historical daily sales records as the forecasting feature.

### Implementation

1. **Data Loading and Preprocessing**
   - Loaded the Rossmann dataset using Pandas.
   - Converted the Date column into datetime format.
   - Selected sales records for a single store.
   - Sorted records chronologically.
   - Set the date as the DataFrame index and handled missing observations.

2. **Data Splitting and Normalization**
   - Divided the dataset into 80% training data and 20% testing data.
   - Used chronological splitting to prevent future data leakage.
   - Normalized sales values using the maximum sales value from the training dataset.

3. **Sequence Generation**
   - Implemented a sliding-window approach to prepare time-series data.
   - Used the previous 30 consecutive days of sales to predict the next day's sales.
   - Reshaped the input data into the 3D format required by LSTM:
     `(samples, time_steps, features)`.

4. **LSTM Model Development**
   - Built a Sequential neural network using TensorFlow and Keras.
   - Added an LSTM layer with 32 units.
   - Added a Dense output layer with one neuron for next-day sales prediction.
   - Compiled the model using the Adam optimizer and Mean Squared Error (MSE) loss function.

5. **Model Training**
   - Trained the model for 20 epochs.
   - Used a batch size of 32.
   - Disabled shuffling to maintain the chronological order of training sequences.

6. **Model Evaluation**
   - Generated predictions on the testing dataset.
   - Calculated Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
   - Compared actual and predicted sales values.
   - Visualized the results to inspect forecasting performance.

7. **Model Saving and Forecasting**
   - Saved the trained model in Keras format as `demand_lstm.keras`.
   - Used the latest 30 consecutive days of sales data to generate a next-day sales forecast.
   - Prepared the forecasting module for future integration with RetailMind's inventory planning functionality.

### Model Architecture


Input Layer (30 time steps, 1 feature)
              |
         LSTM (32 Units)
              |
          Dense (1)
              |
     Next-Day Sales Prediction


### Technologies Used
- Python
- Pandas
- TensorFlow
- Keras
- Matplotlib


### Output
- Trained LSTM demand forecasting model.
- Actual versus predicted sales comparison.
- MAE and RMSE evaluation metrics.
- Next-day store sales prediction.
- Saved model: `models/demand_lstm.keras`


## Day 10 – Seq2Seq & Attention-Based Demand Forecasting

### Objective
Extended the Day 9 LSTM demand forecasting model to predict multiple future days using a Sequence-to-Sequence (Seq2Seq) architecture with an Attention mechanism.

### Dataset
- Dataset: Rossmann Store Sales
- Store selected: Store 1
- Target variable: Sales

### Implementation

1. **Data Preparation**
   - Loaded the Rossmann sales dataset.
   - Selected sales records for Store 1.
   - Converted the Date column into datetime format.
   - Sorted sales records chronologically.
   - Applied an 80:20 chronological train-test split.
   - Normalized sales using the maximum training sales value.

2. **Sequence Generation**
   - Used the previous 30 days as input.
   - Generated the next 14 days as target output.
   - Implemented a sliding-window approach to create training and testing sequences.

3. **Seq2Seq Model Development**
   - Implemented an Encoder-Decoder LSTM architecture.
   - Used the Encoder LSTM to learn historical sales patterns.
   - Used RepeatVector to prepare the decoder input for 14 future time steps.
   - Implemented an Attention mechanism to identify relevant historical information.
   - Used a Dense output layer to generate 14-day sales predictions.

4. **Model Training**
   - Optimizer: Adam
   - Loss Function: Mean Squared Error (MSE)
   - Evaluation Metric: Mean Absolute Error (MAE)
   - Used EarlyStopping to reduce unnecessary training and help prevent overfitting.

5. **Model Evaluation**
   - Evaluated the Seq2Seq model using MAE and RMSE.
   - Compared its performance with the Day 9 single-step LSTM model.
   - Used recursive forecasting with the Day 9 model to generate comparable 14-day predictions.

6. **Attention Visualization**
   - Extracted attention weights from the trained model.
   - Created a heatmap to visualize how the decoder focuses on historical time steps while predicting future sales.

7. **Future Demand Forecasting**
   - Generated sales predictions for the next 14 days.
   - Visualized historical sales alongside predicted sales.
   - Exported future predictions into a CSV file.

### Model Architecture

**Input:** Previous 30 days of sales

**Architecture:**
- Encoder LSTM (32 units)
- Decoder LSTM (32 units)
- Attention Mechanism
- Concatenation Layer
- Dense Output Layer

**Output:** Next 14 days of predicted sales

### Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

### Files Generated

| File | Description |
|---|---|
| `10_seq2seq_attention_forecasting.ipynb` | Day 10 implementation notebook |
| `demand_seq2seq_attention.keras` | Trained Seq2Seq Attention model |
| `demand_forecast_14_days.csv` | 14-day future sales predictions |
| `day9_day10_comparison.csv` | Performance comparison between Day 9 and Day 10 |

### Key Learnings
- Understanding Sequence-to-Sequence learning.
- Understanding Encoder-Decoder LSTM architecture.
- Implementing Attention mechanisms in time-series forecasting.
- Performing multi-step sales forecasting.
- Evaluating forecasting models using MAE and RMSE.
- Visualizing attention weights to understand model focus.

### Outcome
Successfully implemented an Attention-based Seq2Seq forecasting pipeline to predict 14 days of future sales and compare its performance against the Day 9 LSTM baseline.
