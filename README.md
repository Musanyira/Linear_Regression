# House Price Prediction - Machine Learning AI Application

This project is focused on developing and deploying a machine learning-based AI application that predicts house prices. It uses both **Linear Regression** and **Decision Tree** models to predict house prices based on various features such as the number of bedrooms, square footage, and more.

## Project Structure

- **ml_project.ipynb**: Jupyter Notebook containing the following:
  - Data Preprocessing
  - Model Training (Linear Regression and Decision Tree)
  - Model Evaluation and Comparison
  - Saving the Best Model
  - Visualizations of Results

- **app.py**: Streamlit application that:
  - Accepts user input for house features
  - Loads the trained model
  - Displays house price predictions

- **best_house_price_model.pkl**: The saved best-performing machine learning model (Linear Regression or Decision Tree).

- **scaler.pkl**: The saved **StandardScaler** used for feature scaling.

## Requirements

To run this project, you'll need to install the following dependencies:

- Python 3.x
- Streamlit
- scikit-learn
- joblib
- pandas
- numpy
- matplotlib
- seaborn

You can install these dependencies by running:

```bash
