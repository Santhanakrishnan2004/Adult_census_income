
# 💰 Adult Census Income Prediction

This project predicts whether a person earns **>50K** or **<=50K** annually using the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/adult).  
It uses **Scikit-learn**, **Pandas**, and a **Streamlit web app** for deployment.

---

## 🚀 Features
- End-to-end ML pipeline with:
  - Data preprocessing (missing values, categorical encoding, scaling)
  - Model training & evaluation (classification report, ROC-AUC, confusion matrix)
- Visualizations:
  - Confusion Matrix heatmap
  - ROC Curve
  - Precision-Recall Curve
  - Feature Importance
- Interactive **Streamlit App** to predict income from user inputs
- Exported trained model (`clf.pkl`) with preprocessing included

---

## 🛠️ Installation
Clone the repository:
```bash
git clone https://github.com/your-username/adult-income-predictor.git
cd adult-income-predictor
````

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # for Linux/Mac
.venv\Scripts\activate      # for Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Train the model (optional, already trained model provided)

```bash
python train.py
```

This will generate `clf.pkl`.

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Example Outputs

* **Confusion Matrix Heatmap**
  Shows correct vs incorrect classifications.
* **ROC-AUC**
  Measures overall model discrimination ability.
* **Feature Importance**
  Highlights which features most affect predictions.

---

## 📦 Project Structure

```
├── app.py              # Streamlit app
├── train.py            # Model training script
├── clf.pkl             # Saved trained model
├── requirements.txt    # Dependencies
├── README.md           # Project documentation
└─  adult.csv       # Dataset
```

---

## 🌐 Deployment

* **Local** → Run with `streamlit run app.py`
* **Streamlit Cloud** → Free 1-click deploy
* **Hugging Face Spaces** → Works with Streamlit or Gradio
* **Docker** → For AWS/GCP/Azure deployment

---

## 📌 Requirements

* Python 3.9+
* Libraries:

  * scikit-learn
  * pandas
  * numpy
  * matplotlib
  * seaborn
  * streamlit
  * joblib

---

## ✨ Acknowledgements

* Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/adult)
* Libraries: Scikit-learn, Streamlit, Pandas, Numpy

```

