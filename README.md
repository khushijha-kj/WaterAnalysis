# 💧 Smart Water Reuse Predictor

A Machine Learning-powered system to determine the best reuse application of white water, developed as part of a sustainable water management initiative.

---

## 🌱 Project Overview

Once gray water is filtered and cleaned, it becomes *white water*. Although not potable, white water can be safely reused in various non-drinking applications such as irrigation, industrial use, and domestic cleaning. But how do we decide its ideal use case?

This project leverages **Machine Learning** to predict the most sustainable way to reuse white water based on its lab-tested parameters.

---

## 🧪 Dataset

The dataset consists of manually collected water samples that were tested in the lab. Each sample includes parameters such as:

- pH
- Turbidity
- TDS (Total Dissolved Solids)
- BOD (Biological Oxygen Demand)
- COD (Chemical Oxygen Demand)
- Temperature
- and more...

These parameters help in identifying patterns and determining the most suitable use case for a given sample.

---

## 🤖 Machine Learning Workflow

1. **Data Collection & Labeling**  
   Lab-tested white water samples were collected and categorized based on potential reuse applications.

2. **Clustering**  
   We applied **Hierarchical Clustering** to group similar water profiles together, helping us identify natural clusters of use cases.

3. **Supervised Learning**  
   Using the labeled data, we trained a **Support Vector Machine (SVM)** classifier to predict the reuse category of new water samples.

4. **Model Integration**  
   The trained model was saved and integrated into a web application that accepts new water parameters and returns the predicted reuse application.

---

## 📲 Features

- Accepts input water parameters via a user-friendly app interface.
- Uses a trained SVM model to give accurate use case predictions.
- Promotes sustainable and responsible water reuse.

---

## 🔧 Technologies Used

- Python (Pandas, Scikit-learn, Matplotlib)
- Jupyter Notebook
- Machine Learning (Clustering + SVM)
- Web Framework (Flask/Streamlit – if applicable)
- Git & GitHub for version control

---

## 🚀 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/khushijha-kj/WaterAnalysis.git
   cd smart-water-reuse-predictor
   ```
