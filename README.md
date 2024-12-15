# Stock Trading System

This repository contains a reinforcement learning-based stock trading system. The project includes implementations of various Deep Q-Network (DQN) models and their variants, along with evaluation metrics and visualization tools.

## Features
- **Deep Reinforcement Learning Models**: Implementations of DQN, Double DQN, Dueling DQN, and DDDQN.
- **Evaluation Metrics**: Tools for calculating portfolio value, Sharpe ratio, max drawdown, and other key metrics.
- **Data Handling**: Preprocessing tools for historical stock data.
- **Visualization**: Scripts for generating insightful plots like action distributions and portfolio value charts.

## Project Structure
- `agents/`: Contains implementations of reinforcement learning models.
- `data/`: Historical stock market data for training and evaluation.
- `logs/`: Training and evaluation logs in NPZ and log formats.
- `plots/`: Generated visualizations for training and evaluation results.
- `saved_models/`: Pre-trained model weights.
- `utils.py`: Utility functions for data processing and metrics calculation.
- `train.py`: Training script for the reinforcement learning models.
- `evaluate.py`: Evaluation script for testing trained models.

## Getting Started
### Prerequisites
Install the necessary Python dependencies using the provided `requirements.txt`.

pip install -r requirements.txt

### Training
Train a model using the train.py script:

python train.py --model_name DQN --stock_name SPX_2000_2023 --num_episode 10


### Evaluation
Evaluate a trained model using the evaluate.py script:

python evaluate.py --model_path saved_models/DQN_best.pth --data data/SPX_2024.csv

Other options: DoubleDQN_best, DuelingDQN_best, DDDQN_best


### Visualizations

Generated plots can be found in the plots/ directory after running the evaluation script.


