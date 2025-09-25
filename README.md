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

**Target:**

- `Price` – Median house value for California districts (in 100,000s USD).

---

## ⚙️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib
