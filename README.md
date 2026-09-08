# 🚀 Machine Learning Churn Prediction Pipeline

This project is a foundational Python application that generates synthetic data from scratch and executes an end-to-end machine learning pipeline to predict customer churn.

## 📊 Project Overview & Dataset

Instead of importing an external, ready-made dataset, this project uses the `numpy` library to generate exactly 150 rows of completely randomized synthetic customer data. To create a realistic scenario, the target variable `churn` is distributed as 70% "retained" (0) and 30% "churned" (1).

| Feature | Description |
| :--- | :--- |
| **yas (age)** | Randomly generated customer age between 18 and 70. |
| **gelir (income)** | Randomly assigned annual income between 30,000 and 150,000. |
| **abonelik_suresi (subscription_duration)** | Total membership duration ranging from 1 to 60 months. |
| **destek_talebi_sayisi (support_ticket_count)** | Number of customer service interactions ranging from 0 to 10. |
| **sehir (city)** | One of the following values: Istanbul, Ankara, Izmir, or Bursa. |
| **uyelik_tipi (membership_type)** | Standard or Premium membership status. |
| **churn (Target)** | Indicates whether the customer churned (1) or was retained (0). |

## ⚙️ Data Preprocessing Steps

The raw dataset undergoes the following stages to be transformed into a mathematical format that machine learning algorithms can process:

* **Feature Engineering:** A new binary feature named `destek_talebi_var_mi` (0 or 1) is derived based on the total number of support tickets to indicate whether the customer has ever requested support.
* **Missing Value Check:** Before processing, the dataset is checked for null values, and the result is printed to the screen.
* **Categorical Data Transformation (One-Hot Encoding):** Text-based categorical columns like `sehir` and `uyelik_tipi` are converted into a numerical format (1s and 0s) using the `pd.get_dummies` function. The `drop_first=True` parameter is used to prevent the dummy variable trap and avoid data redundancy.
* **Scaling:** To prevent machine learning algorithms from being disproportionately affected by variables with large numbers (e.g., the income column), numerical columns are brought to the same scale using `StandardScaler`.

## 🔀 Model Training and Automatic Selection

Instead of training just a single model, two different algorithms compete in this project, and the system dynamically selects the most successful one:

* **Data Splitting (Stratified Split):** The dataset is split into Train, Validation, and Test sets while maintaining the class imbalance in the target variable (`stratify=y`).
* **Algorithms:** `LogisticRegression` and `KNeighborsClassifier` (KNN, n_neighbors=5) algorithms are trained simultaneously on the training set.
* **Automatic Decision Mechanism:** The Accuracy performances of both models on the Validation set are compared. Whichever algorithm scores higher is automatically assigned (`secilen_model_adi`) as the final model to be used in the testing phase.

## 📈 Testing and Performance Metrics

The winning model from the validation phase is evaluated on the final Test data, which it has never seen before, and a detailed results scorecard is printed to the terminal screen:

* **Confusion Matrix:** Displays a table listing the correct predictions and the errors (False Positives/Negatives) made by the model.
* **Core Metrics:** Accuracy, Precision, Recall, and F1-Score performance values are calculated. (The `zero_division=0` parameter is added to the calculations to prevent crashes during division-by-zero scenarios).
* **Result Interpretation:** At the very end of the console output, an automated comment block explains why the specific model was selected and evaluates the overall pipeline performance based on the synthetic data.

## 💻 Installation and Execution

Follow these steps to run the project in your own environment:

1. Download or clone this repository to your computer.
2. Open your terminal or command prompt and ensure the required libraries are installed using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python file via the terminal or your preferred IDE (e.g., VS Code):
   ```bash
   python churn_prediction.py
  ```
