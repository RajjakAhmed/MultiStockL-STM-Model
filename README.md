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

