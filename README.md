# 🏙️ Munich Real Estate Price Prediction

## 📌 Project Overview
This project aims to predict the base rent (`baseRent`) of apartments in Munich, Germany, using Machine Learning. It serves as an end-to-end practical application of Data Cleaning, Exploratory Data Analysis (EDA), Feature Engineering, and Model Training using `pandas` and `scikit-learn`.

## 📊 Dataset
The data originates from the popular "Apartment rental offers in Germany" dataset (scraped from ImmoScout24) available on Kaggle. The raw dataset was filtered to exclusively include listings in **Munich (München)**.

## 🛠️ Feature Engineering & Data Cleaning
- **Handling Outliers:** Removed unrealistic entries (e.g., a 60,000 sqm house, €20,000 tiny rooms) to prevent the models from learning incorrect patterns ("Garbage in, Garbage out").
- **Feature Selection:** Focused on highly predictive variables:
  - Space & Layout: `livingSpace`, `noRooms`
  - Condition: `condition` (One-Hot Encoded)
  - Amenities: `Balcony`, `hasKitchen`, `lift`
  - Location: Munich districts `regio3` (One-Hot Encoded)
- **Preprocessing:** Applied `StandardScaler` to numerical features after a rigorous 80/20 Train-Test split to prevent data leakage.

## 🚀 Models & Results
I iterated on the models in two phases. Phase 1 used basic features (space, rooms, condition). Phase 2 introduced location and amenities, which drastically improved the performance.

| Model                  | RMSE (Phase 1) | RMSE (Phase 2 - Optimized) |
|------------------------|----------------|----------------------------|
| **Linear Regression**  | 358.11 €       | 290.00 €                   |
| **Random Forest**      | 361.44 €       | 308.00 €                   |
| **Gradient Boosting**  | -              | **286.20 €** 🏆           |

*Note: RMSE (Root Mean Squared Error) represents the average error in Euros when predicting the monthly rent.*

## 💡 Key Learnings
1. **Data Quality > Algorithm Complexity:** Introducing the exact district and basic amenities dropped the error by ~70€ across all models. Feature engineering was the real game-changer.
2. **Baseline Strength:** Linear Regression performed incredibly well, proving that the Munich rental market follows highly logical, linear rules based on size and location.
3. **The Power of Ensembles:** With the increased dimensionality of the Phase 2 dataset, Gradient Boosting outperformed the others by iteratively correcting its own errors.

## 💻 How to Run
1. Clone this repository.
2. Download the dataset from Kaggle and place `immoscout_data.csv` in the root folder.
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
4. Run the Jupyter Notebook `munich_analysis.ipynb`.