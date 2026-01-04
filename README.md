# 🏠 House Prices Prediction using TensorFlow Decision Forests (TF-DF)

This repository implements a **house price prediction system** using **TensorFlow Decision Forests (TF-DF)**. The project is part of **PRODIGY ML Internship – Task 01** and focuses on regression using tabular data.

---

## 📌 Project Overview

The objective is to predict the **SalePrice** of houses based on various numerical and categorical features from the Kaggle *House Prices* dataset.

**Why TF-DF?**

* Excellent performance on tabular data
* Handles categorical features automatically
* No feature scaling required
* Simple training pipeline

---

## 🧠 Model Details

* **Algorithm:** Random Forest Regressor
* **Framework:** TensorFlow Decision Forests
* **Task:** Regression
* **Label Column:** `SalePrice`

---

## 📂 Repository Structure

```
PRODIGY_ML_01/
│
├── house-prices-prediction-using-tfdf.ipynb
├── train.csv
├── test.csv
├── submission.csv
└── README.md
```

---

## ⚙️ Installation

```bash
pip install tensorflow tensorflow-decision-forests pandas scikit-learn
```

---

## 🚀 How to Run

1. Open `house-prices-prediction-using-tfdf.ipynb`
2. Run all cells sequentially
3. Model trains and generates `submission.csv`

---

## 📊 Output Preview

| Id   | SalePrice |
| ---- | --------- |
| 1461 | 208500    |
| 1462 | 181500    |
| 1463 | 223500    |

---

## 🏁 Conclusion

TensorFlow Decision Forests provide a powerful yet simple approach for regression on structured data. This project demonstrates an end-to-end ML workflow with minimal preprocessing.

---

## 👨‍💻 Author

**Akash S**
Machine Learning Intern – Prodigy InfoTech

⭐ If you found this useful, please star the repository!
