import argparse
import importlib
import logging
import sys
import torch
import numpy as np

from utils import *
parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_to_load', action="store", dest="model_to_load", default='DoubleDQN_best', help="model name to load (without .pth)")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='SPX_2023', help="stock name")
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int, help='initial balance')
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int, help='window_size')
inputs = parser.parse_args()

model_to_load = inputs.model_to_load
model_name = model_to_load.split('_')[0]
stock_name = inputs.stock_name
initial_balance = inputs.initial_balance
window_size = inputs.window_size
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for evaluation")

model_module = importlib.import_module(f'agents.{model_name}')
agent = model_module.Agent(state_dim=window_size+3, balance=initial_balance, is_eval=True, model_name=model_to_load)
agent.to(device)

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1

def hold():
    logging.info('Hold')

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        agent.buy_dates.append(t)
        msg = 'Buy:  ${:.2f}'.format(stock_prices[t])
        logging.info(msg)

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        agent.sell_dates.append(t)
        msg = 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)
        logging.info(msg)

log_filename = f'logs/{model_name}_evaluation_{stock_name}.log'
logging.basicConfig(filename=log_filename, filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Evaluating Model:         {model_name}')
logging.info(f'Stock:                    {stock_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

print(f"Evaluating {model_name} on {stock_name}...")
print(f"Trading Period: {trading_period} days, Window Size: {window_size}")

agent.reset_portfolio()
state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
state = torch.tensor(state, dtype=torch.float, device=device)
action_counts = {0: 0, 1: 0, 2: 0}
portfolio_values = []
daily_returns = []
for t in range(1, trading_period + 1):
    if model_name == 'PPO':
        action, log_prob = agent.act(state)
    elif model_name == 'DDPG':
        continuous_action = agent.act(state)
        action = np.clip(int(continuous_action * 3), 0, 2)
    else:
        actions = agent.predict(state)
        action = agent.act(state)

    if action == 1 and agent.balance < stock_prices[t]:
        logging.warning(f"Action 'Buy' at step {t} is invalid due to insufficient balance. Changing to 'Hold'.")
        action = np.random.choice([0, 2])
    if action == 2 and len(agent.inventory) == 0:
        logging.warning(f"Action 'Sell' at step {t} is invalid due to no holdings. Changing to 'Hold'.")
        action = 0

    action_counts[action] += 1
    if action == 0:
        hold()
    if action == 1 and agent.balance > stock_prices[t]:
        buy(t)
    if action == 2 and len(agent.inventory) > 0:
        sell(t)

    next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
    next_state = torch.tensor(next_state, dtype=torch.float, device=device)

    previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
    current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
    delta = current_portfolio_value - previous_portfolio_value

    if previous_portfolio_value > 0:
        daily_return = delta / previous_portfolio_value
        if abs(daily_return) > 1.0:
            logger.warning(f"Daily return is too large: {daily_return}")
            daily_return = 0.0
        agent.return_rates.append(daily_return)
    else:
        agent.return_rates.append(0.0)
        logger.warning(f"Previous portfolio value is non-positive: {previous_portfolio_value}")

    agent.portfolio_values.append(current_portfolio_value)
    state = next_state

    action_counts[action] += 1

    portfolio_values.append(current_portfolio_value)

    if t > 1:
        daily_return = (current_portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
        daily_returns.append(daily_return)

    if t == trading_period:
        portfolio_return= evaluate_portfolio_performance(agent, logging)
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        max_drawdown = calculate_max_drawdown(portfolio_values)
        print(f"Portfolio_return {portfolio_return:.2f}, Sharpe_ratio: {sharpe_ratio: .2f}, max_drawdown: {max_drawdown: .2f}")
        print(f"Evaluation Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
        logging.info(f"Evaluation Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")

np.savez(f'logs/{model_name}_evaluation_data.npz',
         action_counts=action_counts,
         portfolio_values=portfolio_values,
         daily_returns=daily_returns)

plot_all(stock_name, agent, model_to_load)
plot_evaluate_portfolio_value(portfolio_values, model_to_load, stock_name)
plot_evaluate_daily_returns(daily_returns, model_to_load, stock_name)
