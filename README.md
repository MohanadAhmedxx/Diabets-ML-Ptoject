# Diabetes Prediction Using Machine Learning

## Project Overview

This project uses Machine Learning classification algorithms to predict whether a person is diabetic or non-diabetic based on medical diagnostic measurements.

The project applies feature scaling and compares two classification models:

- Support Vector Classifier (SVC) with a linear kernel
- Logistic Regression

The models are evaluated using classification accuracy on both the training and testing datasets.

---

### Dataset

The dataset contains 768 samples and 9 columns.

It consists of 8 input features and 1 target variable ("Outcome").

### Features

The model uses the following features:

| Feature | Description
|---|---|
|"Pregnancies"| Number of times the patient has been pregnant
|"Glucose"| Plasma glucose concentration
|"BloodPressure"| Diastolic blood pressure
|"SkinThickness"| Triceps skin fold thickness
|"Insulin"| 2-Hour serum insulin
|"BMI"| Body Mass Index
|"DiabetesPedigreeFunction"| Diabetes pedigree function, representing the genetic influence of diabetes
|"Age"| Age of the patient

### Target

"Outcome" is the target variable:

- "0" → Non-diabetic
- "1" → Diabetic

---

### Libraries Used

The project uses the following Python libraries:

- NumPy – numerical operations
- Pandas – data loading and data manipulation
- Scikit-learn – data preprocessing, dataset splitting, machine learning models, and evaluation

### Scikit-learn Modules

The following tools from Scikit-learn are used:

StandardScaler
SVC
train_test_split
accuracy_score
LogisticRegression

---

Machine Learning Workflow

### The project follows these main steps:

#### 1. Import the Required Libraries

The required Python and Scikit-learn modules are imported.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

---

#### 2. Load the Dataset

The diabetes dataset is loaded using Pandas:

data = pd.read_csv("diabetes.csv")

The dataset contains:

768 rows × 9 columns

---

#### 3. Separate Features and Target

The "Outcome" column is separated from the input features.

X = data.drop(columns="Outcome", axis=1)
Y = data["Outcome"]

Therefore:

- "X" contains 8 features
- "Y" contains the target variable

The feature matrix has the dimension:

X → (768, 8)

and the target has:

Y → (768,)

---

#### 4. Split the Dataset

The dataset is divided into training and testing sets using "train_test_split".

x_train, x_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    stratify=Y,
    random_state=5
)

The test size is 30% of the dataset.

The resulting dimensions are:

Training set: (537, 8)
Testing set:  (231, 8)

Therefore:

- 537 samples are used for training.
- 231 samples are used for testing.
- Both sets contain the same 8 input features.

"stratify=Y" is used to preserve approximately the same class distribution in the training and testing sets.

---

#### 5. Feature Scaling

Since the features have different numerical ranges, "StandardScaler" is used to standardize the training features.

Scale = StandardScaler()

Scale.fit(x_train)

x_train = Scale.transform(x_train)

Standardization transforms the features so that they have approximately:

- Mean = 0
- Standard deviation = 1

After scaling, the training data still has the same dimensions:

x_train → (537, 8)

---

Model 1: Linear Support Vector Classifier

The first model is a Support Vector Classifier with a linear kernel.

model_1 = SVC(kernel="linear")
model_1.fit(x_train, y_train)

The model learns a linear decision boundary that separates the two classes:

- Diabetic
- Non-diabetic

Training Accuracy

The trained model is used to predict the training data:

x_train_prediction = model_1.predict(x_train)

The training accuracy obtained in the notebook is approximately:

79.70%

Test Accuracy

The model is then evaluated on the test set.

The notebook obtains approximately:

74.89%

---

Model 2: Logistic Regression

The second classification algorithm used in the project is Logistic Regression.

from sklearn.linear_model import LogisticRegression

model_2 = LogisticRegression()

model_2.fit(x_train, y_train)

The model is trained using the same scaled training features.

Training Accuracy

The training accuracy obtained is approximately:

79.70%

Test Accuracy

The test accuracy obtained in the notebook is approximately:

75.76%

---

Results

The obtained accuracy values are:

Model| Training Accuracy| Testing Accuracy
Linear SVM| 79.70%| 74.89%
Logistic Regression| 79.70%| 75.76%

The results show that both models achieved similar performance on the training data, while their testing accuracies were slightly different.

---

### Complete Workflow

The complete Machine Learning workflow can be summarized as:

Diabetes Dataset
       ↓
Load Dataset with Pandas
       ↓
Explore Dataset
       ↓
Separate Features and Target
       ↓
X = 8 Features
Y = Outcome
       ↓
Train/Test Split
       ↓
537 Training Samples
231 Testing Samples
       ↓
Standardize Training Features
       ↓
Train Linear SVM
       ↓
Evaluate Training & Testing Accuracy
       ↓
Train Logistic Regression
       ↓
Evaluate Training & Testing Accuracy
       ↓
Compare Results

---

### Conclusion

This project demonstrates a basic Machine Learning classification workflow for diabetes prediction.

Two classification algorithms were implemented:

1. Linear Support Vector Classifier (SVC)
2. Logistic Regression

Both models use the same eight medical features and are trained after applying feature standardization.

The final testing accuracies obtained in this notebook are approximately 74.89% for Linear SVM and 75.76% for Logistic Regression.

The project demonstrates the complete process from loading and preparing the dataset to training, prediction, and model evaluation.