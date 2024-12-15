import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from utils import *
from utils import calculate_sharpe_ratio, calculate_max_drawdown

parser = argparse.ArgumentParser(description='command line options')
parser.add_argument('--model_name', action="store", dest="model_name", default='DQN', help="model name")
parser.add_argument('--stock_name', action="store", dest="stock_name", default='SPX_2000_2022', help="stock name")
parser.add_argument('--window_size', action="store", dest="window_size", default=10, type=int,
                    help="span (days) of observation")
parser.add_argument('--num_episode', action="store", dest="num_episode", default=10, type=int, help='episode number')
parser.add_argument('--initial_balance', action="store", dest="initial_balance", default=50000, type=int,
                    help='initial balance')
inputs = parser.parse_args()

model_name = inputs.model_name
stock_name = inputs.stock_name
window_size = inputs.window_size
num_episode = inputs.num_episode
initial_balance = inputs.initial_balance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

stock_prices = stock_close_prices(stock_name)
trading_period = len(stock_prices) - 1
returns_across_episodes = []
num_experience_replay = 0
action_dict = {0: 'Hold', 1: 'Buy', 2: 'Sell'}

model = importlib.import_module(f'agents.{model_name}')
agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)
agent.to(device)

Path("logs").mkdir(parents=True, exist_ok=True)
Path("saved_models").mkdir(parents=True, exist_ok=True)

def hold(actions_np, t):
    next_probable_action = np.argsort(actions_np)[1]
    if next_probable_action == 2 and len(agent.inventory) > 0:
        max_profit = stock_prices[t] - min(agent.inventory)
        if max_profit > 0:
            sell_msg = sell(t)
            actions_np[next_probable_action] = 1
            return 'Hold', actions_np
    return 'Hold', actions_np

def buy(t):
    if agent.balance > stock_prices[t]:
        agent.balance -= stock_prices[t]
        agent.inventory.append(stock_prices[t])
        return 'Buy: ${:.2f}'.format(stock_prices[t])
    return None

def sell(t):
    if len(agent.inventory) > 0:
        agent.balance += stock_prices[t]
        bought_price = agent.inventory.pop(0)
        profit = stock_prices[t] - bought_price
        global reward
        reward = profit
        return 'Sell: ${:.2f} | Profit: ${:.2f}'.format(stock_prices[t], profit)
    return None

logging.basicConfig(filename=f'logs/{model_name}_training_{stock_name}.log', filemode='w',
                    format='[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Trading Object:           {stock_name}')
logging.info(f'Trading Period:           {trading_period} days')
logging.info(f'Window Size:              {window_size} days')
logging.info(f'Training Episode:         {num_episode}')
logging.info(f'Model Name:               {model_name}')
logging.info('Initial Portfolio Value: ${:,}'.format(initial_balance))

print(f"Training {model_name} on {stock_name} for {num_episode} episodes...")
print(f"Window Size: {window_size}, Trading Period: {trading_period} days")

start_time = time.time()

best_return = -np.inf
best_stable_return= -np.inf
best_min_drawdown= np.inf

action_distributions = []
portfolio_values = []
daily_returns = []
sharpe_ratios = []
max_drawdowns = []
rewards_per_episode = []

for e in range(1, num_episode + 1):
    logging.info(f'\nEpisode: {e}/{num_episode}')
    print(f"\n=== Episode {e}/{num_episode} ===")

    agent.reset()
    episode_rewards = 0
    state = generate_combined_state(0, window_size, stock_prices, agent.balance, len(agent.inventory))
    state = torch.tensor(state, dtype=torch.float, device=device)
    action_counts = {0: 0, 1: 0, 2: 0}
    with trange(1, trading_period + 1, desc=f"Episode {e}", leave=False) as t_range:
        for t in t_range:
            reward = 0
            current_price = stock_prices[t]
            agent.set_current_price(current_price)
            next_state = generate_combined_state(t, window_size, stock_prices, agent.balance, len(agent.inventory))
            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            previous_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance

            actions = agent.predict(state)
            actions = actions.squeeze(0)
            action = agent.act(state)
            action_counts[action] += 1

            execution_result = None
            actions_np = actions.detach().cpu().numpy().copy()
            if action == 0:
                execution_result = hold(actions_np, t)
                if len(agent.inventory) == 0:
                    reward -= 0.8
                elif action_counts[0] > (t * 0.5):
                    reward -= 0.5
                elif action_counts[0] < (t * 0.3):
                    reward += 0.2
                else:
                    reward -= 0.1 * len(agent.inventory) * stock_prices[t]
            elif action == 1:
                execution_result = buy(t)
                reward += 0.05 * stock_prices[t]
            elif action == 2:
                execution_result = sell(t)
                reward += 0.1 * (stock_prices[t] - min(agent.inventory)) if agent.inventory else 0
            elif execution_result is None:
                reward -= 0.01 * len(agent.inventory) * stock_prices[t]
                missed_profit = max(0, stock_prices[t] - min(agent.inventory)) if agent.inventory else 0
                reward -= missed_profit * 0.01

            if len(agent.portfolio_values) > 1:
                prev_value = agent.portfolio_values[-2]
                if current_portfolio_value > prev_value:
                    reward += 0.5 * (current_portfolio_value - prev_value) / prev_value
                else:
                    reward -= 0.4 * (prev_value - current_portfolio_value) / prev_value

            if action == 0:
                next_actions = agent.predict(next_state).detach().cpu().numpy()
                next_action = np.argmax(next_actions)
                if len(agent.inventory) == 0 and next_action == 1:
                    reward += 0.05
                elif len(agent.inventory) > 0 and next_action == 2:
                    reward += 0.05

            current_portfolio_value = len(agent.inventory) * stock_prices[t] + agent.balance
            delta = current_portfolio_value - previous_portfolio_value
            reward += delta / agent.initial_portfolio_value
            portfolio_values.append(current_portfolio_value)
            if t > 1:
                daily_return = (current_portfolio_value - portfolio_values[-2]) / portfolio_values[-2]
                daily_returns.append(daily_return)

            if len(daily_returns) > 1:
                sharpe_ratio = calculate_sharpe_ratio(daily_returns)
                reward += 1 * sharpe_ratio

            agent.portfolio_values.append(current_portfolio_value)
            agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)

            max_drawdown = maximum_drawdown(agent.portfolio_values)
            if max_drawdown > -0.3:
                reward -= 0.5
            elif max_drawdown > -0.1:
                reward += 0.2

            episode_rewards += reward

            done = True if t == trading_period else False

            agent.remember(state.detach().cpu().numpy(),action,reward,next_state.detach().cpu().numpy(),done)
            state = next_state

            logging.info(
                f"Step {t}: Action={action_dict[action]}, Reward={reward:.2f}, Portfolio Value={current_portfolio_value:.2f}")

            if len(agent.memory) > agent.buffer_size:
                num_experience_replay += 1
                loss = agent.experience_replay()
                info_line = (f"Episode: {e}\tLoss: {loss:.2f}\tAction: {action_dict[action]}"
                             f"\tReward: {reward:.2f}\tBalance: {agent.balance:.2f}"
                             f"\tNumber of Stocks: {len(agent.inventory)}")
                logging.info(info_line)

            if done:
                agent.adjust_epsilon(e, num_episode)
                portfolio_return= evaluate_portfolio_performance(agent, logging)
                returns_across_episodes.append(portfolio_return)

                sharpe_ratio = calculate_sharpe_ratio(daily_returns)
                sharpe_ratios.append(sharpe_ratio)
                max_drawdown = calculate_max_drawdown(portfolio_values)
                max_drawdowns.append(max_drawdown)
                rewards_per_episode.append(episode_rewards)

                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    action_distribution = [
                        action_counts[0] / total_actions,
                        action_counts[1] / total_actions,
                        action_counts[2] / total_actions
                    ]
                else:
                    action_distribution = [0.0, 0.0, 0.0]
                action_distributions.append(action_counts.copy())

                logging.info(
                    f"Episode {e} Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}Portfolio_return {portfolio_return:.2f}, Sharpe_ratio: {sharpe_ratio: .2f}, max_drawdown: {max_drawdown: .2f}")
                print(f"Episode {e} finished with Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
                print(f"Portfolio_return {portfolio_return:.2f}, Sharpe_ratio: {sharpe_ratio: .2f}, max_drawdown: {max_drawdown: .2f}")

                if e % 5 == 0:
                    agent.save(f'saved_models/{model_name}_ep{e}.pth')

                if portfolio_return > best_return:
                    best_return = portfolio_return
                    agent.save(f'saved_models/{model_name}_best.pth')
                    logging.info(f"New best model saved with return {best_return:.2f}")

                if max_drawdown < best_min_drawdown:
                    best_min_drawdown = max_drawdown
                    agent.save(f'saved_models/{model_name}_low_risk.pth')
                    logging.info(f"Low-risk model saved with max drawdown {best_min_drawdown:.2f}.")

                if len(returns_across_episodes) >= 5:
                    avg_recent_return = np.mean(returns_across_episodes[-5:])
                    if avg_recent_return > best_stable_return:
                        best_stable_return = avg_recent_return
                        agent.save(f'saved_models/{model_name}_stable.pth')
                        logging.info(f"Stable model saved with average return {best_stable_return:.2f}.")

np.savez(f'logs/{model_name}_training_data.npz',
         action_distributions=action_distributions,
         portfolio_values=portfolio_values,
         daily_returns=daily_returns,
         sharpe_ratios=sharpe_ratios,
         max_drawdowns=max_drawdowns,
         rewards_per_episode=rewards_per_episode)
logging.info('total training time: {0:.2f} min'.format((time.time() - start_time) / 60))

plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)

action_array = np.array([
    [dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)]
    for dist in action_distributions
])
plot_action_distribution(action_array, model_name)
plot_portfolio_value(portfolio_values, model_name)
plot_daily_returns(daily_returns, model_name)
plot_sharpe_ratios(sharpe_ratios, model_name)
plot_rewards(rewards_per_episode, model_name)
plot_max_drawdowns(max_drawdowns, model_name)
print("finished")

