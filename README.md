Stock LSTM Model & Prediction Analysis

This repository contains code for training and evaluating LSTM models on historical stock data. It leverages Yahoo Finance data and TensorFlow to build predictive models for multiple stocks, then validates these models by generating predictions and performance graphs.

Table of Contents

Overview

Features

Project Structure

Requirements

Installation

Usage

Graphs and Results

License

Overview

This project consists of two main components:

Model Generation (model_generator.py):

Downloads historical stock data using yfinance.

Preprocesses the data and scales the closing prices.

Trains an LSTM neural network using TensorFlow/Keras to predict stock prices.

Saves trained models in the stock_models directory.

Prediction Analysis (prediction_analysis.py):

Loads the trained models.

Downloads updated stock data.

Makes rolling predictions on a test period and computes evaluation metrics (MAE & RMSE).

Generates performance graphs that compare historical stock prices against predictions.

Saves graphs in the prediction_graphs directory.

Features
Multiple Stock Support: Train and validate models for several major companies (AAPL, AMZN, GOOGL, TSLA, INFY, TM).

Rolling Predictions: Uses a moving window of past data to generate future predictions.

Performance Evaluation: Computes key error metrics, including Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

Visualization: Automatically creates and saves graphs that display historical prices, predictions, and evaluation metrics.

Modular Code: Clean, well-documented code to allow easy customization and extension.


                                        stock_models/             # Directory where trained models are saved
                                       
                                        prediction_graphs/        # Directory where generated graphs are stored
                                        
                                        model_generator.py        # Script for training the LSTM models
                                        
                                        prediction_analysis.py    # Script for validating models and generating prediction graphs
                                        
                                        requirements.txt          # List of required Python packages
                                        
                                        .gitignore                # Git ignore file to exclude unnecessary files


Requirements
The project relies on the following Python libraries:

numpy
pandas
yfinance
matplotlib
scikit-learn
tensorflow

All dependencies are listed in the requirements.txt file.

Installation
          git clone https://github.com/RajjakAhmed/MultiStockL-STM-Model.git
          cd MultiStockL-STM-Model

Create a virtual environment
          python -m venv venv
            source venv/bin/activate  # On Windows use: venv\\Scripts\\activate

Install dependencies:
          pip install -r requirements.txt

Usage
1. Model Training

Run the model generator script to download historical data, preprocess it, and train the LSTM model for each stock.

python model_generator.py

2. Prediction and Validation


After training, run the prediction analysis script to validate the models and generate performance graphs:

python prediction_analysis.py


Predicted performance graphs will be saved in the prediction_graphs directory, and evaluation metrics will be printed to the console.

Graphs and Results


Each time you run prediction_analysis.py, a graph will be generated for each stock that shows:

Historical Prices: Actual closing prices from the dataset.

Predictions: The rolling predictions generated by the model.

Actual Prices: Overlaid actual prices during the test period.

Train/Test Split Indicator: A vertical line indicating the cutoff between training and test data.

Performance Metrics: MAE and RMSE annotated on the graph.

You can add your generated graphs to the repository or share them as needed.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Disclaimer


Important: This project is provided for educational and experimental purposes only. The models generated from this project are not suitable for actual financial or stock market predictions due to:

Simplified Modeling: The models are based solely on historical data and may not capture market complexities.

Market Volatility: Real-world financial markets are influenced by unpredictable external factors not considered in these models.

Risk of Loss: Using these predictions for real trading may result in financial loss.

Always consult a professional financial advisor before making investment decisions.

License


This project is licensed under the MIT License. See the LICENSE file for more details.




Feel free to further modify this README as your project evolves. Happy coding and experimentation!



