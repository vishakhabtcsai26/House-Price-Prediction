# California House Price Prediction

This project demonstrates **house price prediction** using machine learning techniques on the **California Housing Dataset**. The project compares **Linear Regression** and **XGBoost Regressor** models and evaluates their performance using standard metrics.

---

## 📂 Project Structure

- `house_price_prediction.ipynb` – Jupyter Notebook containing the full workflow from data loading, exploration, model training, evaluation, and visualization.
- `xgboost_house_price_model.pkl` – Saved XGBoost model for later use.
- `README.md` – Project overview and instructions.

---

## 📝 Dataset

The project uses the **California Housing Dataset**, available in `scikit-learn`.  

**Features:**

| Feature | Description |
|---------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block group population |
| AveOccup | Average household occupancy |
| Latitude | Block group latitude |
| Longitude | Block group longitude |
<img width="808" height="567" alt="image" src="https://github.com/user-attachments/assets/0d122a1b-a0e9-4814-a9a3-54a1e3037fa6" />
<img width="784" height="576" alt="image" src="https://github.com/user-attachments/assets/aabf2ed0-e3a3-4b75-bcac-0f8146e5a5fe" />


**Target:**

- `Price` – Median house value for California districts (in 100,000s USD).
<img width="847" height="683" alt="image" src="https://github.com/user-attachments/assets/21cc99e6-b4bf-4223-8793-1b0a0bfc263d" />

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
