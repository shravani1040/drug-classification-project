# 💊 Drug Regulatory Classification — ML Web App

A machine learning project that predicts the **regulatory classification of drugs** based on pharmaceutical and clinical attributes. Built with Python, Scikit-learn, and deployed as a Flask web application.

---

## 🎯 Aim

To build a machine learning model that can automatically classify drugs into their correct **Target Regulatory Class** using features like drug form, therapeutic class, abuse potential score, adverse event reports, and more — reducing manual regulatory review effort.

---

## 📁 Project Structure

```
drug-classification-project/
│
├── app.py                  # Flask web application
├── models/
│   ├── model.pkl           # Trained ML model
│   ├── preprocessor.pkl    # Data preprocessing pipeline
│   └── label_encoder.pkl   # Label encoder for target classes
├── templates/
│   └── index.html          # Frontend UI
└── README.md
```

---

## 📊 Dataset

- **Source:** Drug Regulatory Classification Dataset (CSV)
- **Original Size:** 60,000 rows × 30 columns
- **After cleaning:** 57,000 rows (dropped rows with missing target labels)
- **Target Column:** `Target_Regulatory_Class`

### Key Features Used:
| Type | Features |
|------|----------|
| Numerical | `Dosage_mg`, `Abuse_Potential_Score`, `Regulatory_Risk_Score`, `Adverse_Event_Reports`, `Price_Per_Unit`, `Annual_Sales_Volume`, `Recall_History_Count` |
| Categorical | `Drug_Form`, `Therapeutic_Class`, `Manufacturing_Region`, `Requires_Cold_Storage`, `OTC_Flag`, `High_Risk_Substance` |

---

## 🔍 Exploratory Data Analysis (EDA)

- Checked for **missing values** and handled them using median/mode imputation
- Analyzed **class distribution** of the target variable
- Visualized **numerical distributions** using histograms and boxplots
- Studied **correlations** using a heatmap
- Detected and handled **outliers** using Winsorization (IQR method)
- Analyzed **skewness and kurtosis** of numerical features

---

## ⚙️ Preprocessing Pipeline

Built using **Scikit-learn Pipelines** and **ColumnTransformer**:

- **Outlier columns** (`Annual_Sales_Volume`, `Recall_History_Count`, `Adverse_Event_Reports`):
  - Median Imputation → Winsorizer (IQR) → Standard Scaler

- **Normal numerical columns:**
  - Median Imputation → Standard Scaler

- **Categorical columns:**
  - Most-frequent Imputation → One-Hot Encoding

---

## 🤖 Models Trained & Compared

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model, max_iter=1000 |
| K-Nearest Neighbors | k=5, Euclidean distance |
| Support Vector Machine | RBF kernel, trained on 15k sample |
| Artificial Neural Network | MLP (64→32), ReLU, Adam optimizer |
| Decision Tree | Max depth=10, Gini criterion, balanced class weight |

All models were evaluated on an **80/20 stratified train-test split**.

---

## 📈 Results

Models were compared on **accuracy** on the held-out test set. A bar chart comparison was generated to visualize differences between all 5 models.

> The best performing model was saved and deployed in the Flask app.

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/drug-classification-project.git
cd drug-classification-project
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install flask scikit-learn pandas numpy feature-engine
```

### 4. Run the Flask app
```bash
python app.py
```

### 5. Open in browser
```
http://127.0.0.1:5000
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML models & preprocessing |
| Feature-engine | Winsorization / outlier handling |
| Matplotlib & Seaborn | Data visualization |
| Flask | Web application framework |
| Pickle | Model serialization |

---

## ✅ Conclusion

- Successfully built an end-to-end drug regulatory classification pipeline
- Compared 5 ML models to find the best performing one
- Deployed the final model as a **Flask web app** for real-time predictions
- Preprocessing pipeline ensures consistent transformation of new input data

---

## 👤 Author

**Shravani**  
Drug Regulatory Classification ML Project
