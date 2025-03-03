# Neural_Network_Delivery_Time_Prediction

This project focuses on predicting delivery times for orders placed through **Porter**, India's largest marketplace for intra-city logistics. Porter serves over 5 million customers and works with a wide range of restaurants to deliver items directly to customers. The goal of this project is to build a robust regression model that can accurately estimate delivery times based on various features such as order details, restaurant information, and delivery partner availability.

---
# Machine Learning Model Comparison

This project evaluates the performance of **Linear Regression, Neural Networks, and XGBoost** using various metrics.

## 📦 Libraries Used

| Library | Description |
|---------|------------|
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical computing library |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Data manipulation & analysis |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) | Data visualization library |
| ![Seaborn](https://img.shields.io/badge/Seaborn-008080?style=for-the-badge&logo=python&logoColor=white) | Statistical data visualization |
| ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Machine learning library |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) | Deep learning framework |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) | Neural network API |

## 🤖 Models Used

| Model | Description | Image |
|-------|------------|-------|
| **Neural Network** | Deep learning model for complex patterns | ![NN](https://upload.wikimedia.org/wikipedia/commons/e/e4/Artificial_neural_network.svg) |
| **Linear Regression** | Simple yet effective model for linear relationships | ![Linear Regression](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg) |
| **XGBoost** | Gradient boosting model for structured data | ![XGBoost](data:image/png;base64,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) |

## 📊 Model Performance

| Model               | MAE (↓ better) | R² Score (↑ better) |
|---------------------|--------------|------------------|
| **Linear Regression** | 0.2503       | **0.9997**       |
| **Neural Network**   | **0.2097**   | 0.9836           |
| **XGBoost**         | 0.9981       | 0.9746           |

---

---


## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Context](#context)
3. [Dataset](#dataset)
4. [Data Dictionary](#data-dictionary)
5. [Features](#features)
6. [Methodology](#methodology)

---

## **Project Overview**
The project involves:
- **Exploratory Data Analysis (EDA)** to understand the dataset and identify trends.
- **Data preprocessing**, including handling missing values, encoding categorical variables, and removing outliers.
- Building and training a **Neural Network (NN)** model using TensorFlow/Keras.
- Comparing the NN model's performance with **XGBoost** and **Linear Regression**.
- Evaluating models using **Mean Absolute Error (MAE)** and **R? scores**.

---

## **Context**
**Porter** is India's largest marketplace for intra-city logistics, operating in the country's $40 billion intra-city logistics market. The company aims to improve the lives of over 1,50,000+ driver-partners by providing them with consistent earnings and independence. Porter has serviced over 5 million customers and works with a wide range of restaurants to deliver food and other items directly to customers.

To enhance customer experience, Porter wants to provide accurate delivery time estimates to customers based on:
- **What they are ordering** (e.g., item details, subtotal).
- **Where the order is placed** (e.g., market ID, store location).
- **Delivery partner availability** (e.g., on-shift partners, busy partners).

This project uses a dataset containing the necessary features to train a regression model for delivery time estimation.

---

## **Dataset**
The dataset contains information about orders, including:
- **Order details**: `created_at`, `actual_delivery_time`, `subtotal`, `min_item_price`, `max_item_price`, etc.
- **Store details**: `store_id`, `store_primary_category`, `market_id`, etc.
- **Delivery logistics**: `total_onshift_partners`, `total_busy_partners`, `total_outstanding_orders`, etc.

The dataset is stored in a CSV file named `dataset.csv`.

---

## **Data Dictionary**
Each row in the dataset corresponds to one unique delivery. The columns represent the following features:
- **market_id**: Integer ID for the market where the restaurant is located.
- **created_at**: Timestamp at which the order was placed.
- **actual_delivery_time**: Timestamp when the order was delivered.
- **store_primary_category**: Category of the restaurant (e.g., fast food, fine dining).
- **order_protocol**: Integer code representing how the order was placed (e.g., through Porter, call to restaurant, pre-booked, third-party, etc.).
- **total_items**: Total number of items in the order.
- **subtotal**: Final price of the order.
- **num_distinct_items**: Number of distinct items in the order.
- **min_item_price**: Price of the cheapest item in the order.
- **max_item_price**: Price of the costliest item in the order.
- **total_onshift_partners**: Number of delivery partners on duty at the time the order was placed.
- **total_busy_partners**: Number of delivery partners attending to other tasks.
- **total_outstanding_orders**: Total number of orders to be fulfilled at the moment.

---

## **Features**
Key features used in the model:
- **Temporal features**: `day_of_week`, `hour_o`, `minute_o`, etc.
- **Order details**: `subtotal`, `min_item_price`, `max_item_price`.
- **Delivery logistics**: `total_onshift_partners`.
- **Target variable**: `time_taken` (delivery time in minutes).

---

## **Methodology**
1. **Data Preprocessing**:
   - Converted date columns to datetime format.
   - Extracted temporal features (e.g., day of the week, hour of the day).
   - Handled missing values and removed outliers.
   - Applied target encoding to categorical variables.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized the distribution of delivery times.
   - Analyzed order frequencies by hour, day, and market.
   - Explored correlations between features and removed highly correlated columns.

3. **Model Building**:
   - Built a Neural Network model with multiple dense layers, batch normalization, and leaky ReLU activations.
   - Used advanced techniques like learning rate scheduling and early stopping.
   - Trained and evaluated the model using MAE and R².

4. **Comparison with Other Models**:
   - Compared the NN model's performance with XGBoost and Linear Regression.


The **Neural Network model** demonstrated the best performance, with the lowest MAE and a high R² score, indicating strong predictive accuracy and generalization.
---
