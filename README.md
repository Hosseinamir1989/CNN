# CNN-model-pred-stock

This project implements a complete pipeline for predicting stock prices using a Convolutional Neural Network (CNN) and technical indicators. It includes data loading, feature engineering, model training, evaluation, and reporting.

## Features

- Loads historical stock price data for multiple tickers.
- Computes technical indicators (MACD, RSI, Bollinger Bands, etc.).
- Preprocesses and normalizes features.
- Trains a CNN model to predict future stock prices.
- Evaluates model performance (RMSE, MAE, directional accuracy).
- Analyzes feature importance via feature drop analysis.
- Simulates trading strategies and calculates PnL.
- Generates summary and comparison tables for multiple tickers.

## Project Structure

- `main.py` — Main script for running the pipeline.
- `data_loader.py` — Loads and preprocesses stock data.
- `indicators.py` — Adds technical indicators to the data.
- `preprocessing.py` — Normalizes and windowizes data for the CNN.
- `model.py` — Defines and builds the CNN model.
- `evaluation.py` — Evaluation metrics, plotting, and PnL simulation.
- `feature_analysis.py` — Feature drop/importance analysis.
- `evaluation_tables.py` — Generates and saves evaluation tables.
- `download_stock_price.py` — (Optional) Script for downloading raw stock data.
- `config/config.yaml` — Configuration file for tickers, paths, and model parameters.
- `output/` — Output directory for results and evaluation tables.
- `complete_cnn_pipeline_full_reporting.ipynb` — Full pipeline notebook with detailed reporting.
- `presenting_pipeline.ipynb` — Notebook for presenting results and comparisons.

## Usage

1. **Install dependencies**  
   Make sure you have Python 3.8+ and install required packages:
   ```sh
   pip install -r requirements.txt
   ```

2. **Configure the pipeline**  
   Edit `config/config.yaml` to set tickers, data paths, and model parameters.

3. **Run the pipeline**  
   Execute the main script:
   ```sh
   python main.py
   ```
   Or use the Jupyter notebooks for step-by-step execution and visualization.

4. **View results**  
   - Evaluation tables and feature analysis are saved in the `output/` directory.
   - Plots and detailed reports are available in the notebooks.

## Notebooks

- `complete_cnn_pipeline_full_reporting.ipynb`: End-to-end pipeline with full reporting and visualizations.
- `presenting_pipeline.ipynb`: Focused on presenting and comparing results across tickers.

## License

MIT License

---

*For more details, see the code and notebooks in this repository.*