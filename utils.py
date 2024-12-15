import numpy as np
import pandas as pd
from empyrical import sharpe_ratio
from matplotlib import pyplot as plt


class Portfolio:
    def __init__(self, balance=50000):
        self.initial_portfolio_value = balance
        self.balance = balance
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [balance]
        self.buy_dates = []
        self.sell_dates = []

    def reset_portfolio(self):
        self.balance = self.initial_portfolio_value
        self.inventory = []
        self.return_rates = []
        self.portfolio_values = [self.initial_portfolio_value]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def stock_close_prices(key):
    prices = []
    with open("data/" + key + ".csv", "r") as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        prices.append(float(line.split(",")[4]))
    return prices


def generate_price_state(stock_prices, end_index, window_size):
    start_index = end_index - window_size
    if start_index >= 0:
        period = stock_prices[start_index:end_index + 1]
    else:
        pad_length = -start_index
        padded_values = np.full((pad_length,), stock_prices[0])
        period = np.concatenate((padded_values, stock_prices[0:end_index + 1]))
    return sigmoid(np.diff(period))


def generate_portfolio_state(stock_price, balance, num_holding):
    return [np.log(stock_price), np.log(balance), np.log(num_holding + 1e-6)]


def generate_combined_state(end_index, window_size, stock_prices, balance, num_holding):
    price_state = generate_price_state(stock_prices, end_index, window_size)
    portfolio_state = generate_portfolio_state(stock_prices[end_index], balance, num_holding)
    return np.array([np.concatenate((price_state, portfolio_state), axis=None)])


def treasury_bond_daily_return_rate():
    r_year = 2.75 / 100
    return (1 + r_year)**(1 / 365) - 1


def maximum_drawdown(portfolio_values):
    arr = np.array(portfolio_values)
    cumulative_max = np.maximum.accumulate(arr)
    drawdowns = (arr - cumulative_max) / cumulative_max
    return np.min(drawdowns)


def evaluate_portfolio_performance(agent, logger):
    portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    mean_daily_return = np.mean(agent.return_rates) * 100.0 if agent.return_rates else 0
    logger.info('--------------------------------')
    logger.info(f'Portfolio Value:        ${agent.portfolio_values[-1]:.2f}')
    logger.info(f'Portfolio Balance:      ${agent.balance:.2f}')
    logger.info(f'Portfolio Stocks Number: {len(agent.inventory)}')
    logger.info(f'Total Return:           ${portfolio_return:.2f}')
    logger.info(f'Mean/Daily Return Rate:  {mean_daily_return:.3f}%')
    logger.info('--------------------------------')
    return portfolio_return


def buy_and_hold_benchmark(stock_name, agent):
    df = pd.read_csv('./data/{}.csv'.format(stock_name))
    dates = df['Date']
    initial_price = df.iloc[0, 4]
    num_holding = agent.initial_portfolio_value // initial_price
    balance_left = agent.initial_portfolio_value % initial_price
    buy_and_hold_portfolio_values = df['Close'] * num_holding + balance_left
    buy_and_hold_return = buy_and_hold_portfolio_values.iloc[-1] - agent.initial_portfolio_value
    return dates, buy_and_hold_portfolio_values, buy_and_hold_return


def plot_portfolio_transaction_history(stock_name, agent):
    portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    df = pd.read_csv('./data/{}.csv'.format(stock_name))
    buy_prices = [df.iloc[t, 4] for t in agent.buy_dates]
    sell_prices = [df.iloc[t, 4] for t in agent.sell_dates]

    plt.figure(figsize=(15, 5), dpi=100)
    plt.title('{} Total Return on {}: ${:.2f}'.format(agent.model_type, stock_name, portfolio_return))
    plt.plot(df['Date'], df['Close'], color='black', label=stock_name)
    if agent.buy_dates:
        plt.scatter(agent.buy_dates, buy_prices, c='green', alpha=0.5, label='buy')
    if agent.sell_dates:
        plt.scatter(agent.sell_dates, sell_prices, c='red', alpha=0.5, label='sell')
    plt.xticks(np.linspace(0, len(df), 10))
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


def plot_portfolio_performance_comparison(stock_name, agent):
    dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(stock_name, agent)
    agent_return = agent.portfolio_values[-1] - agent.initial_portfolio_value

    plt.figure(figsize=(15, 5), dpi=100)
    plt.title('{} vs. Buy and Hold'.format(agent.model_type))
    plt.plot(dates, agent.portfolio_values, color='green', label='{} Total Return: ${:.2f}'.format(agent.model_type, agent_return))
    plt.plot(dates, buy_and_hold_portfolio_values, color='blue', label='{} Buy and Hold Total Return: ${:.2f}'.format(stock_name, buy_and_hold_return))
    plt.xticks(np.linspace(0, len(dates), 10))
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid()
    plt.show()


def plot_all(stock_name, agent, model_name):
    fig, ax = plt.subplots(2, 1, figsize=(16,8), dpi=100)

    portfolio_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    df = pd.read_csv('./data/{}.csv'.format(stock_name))
    buy_prices = [df.iloc[t, 4] for t in agent.buy_dates]
    sell_prices = [df.iloc[t, 4] for t in agent.sell_dates]
    ax[0].set_title('{} Total Return on {}: ${:.2f}'.format(agent.model_type, stock_name, portfolio_return))
    ax[0].plot(df['Date'], df['Close'], color='black', label=stock_name)
    if agent.buy_dates:
        ax[0].scatter(agent.buy_dates, buy_prices, c='green', alpha=0.5, label='buy')
    if agent.sell_dates:
        ax[0].scatter(agent.sell_dates, sell_prices, c='red', alpha=0.5, label='sell')
    ax[0].set_ylabel('Price')
    ax[0].set_xticks(np.linspace(0, len(df), 10))
    ax[0].legend()
    ax[0].grid()

    dates, buy_and_hold_portfolio_values, buy_and_hold_return = buy_and_hold_benchmark(stock_name, agent)
    agent_return = agent.portfolio_values[-1] - agent.initial_portfolio_value
    ax[1].set_title('{} vs. Buy and Hold'.format(model_name))
    ax[1].plot(dates, agent.portfolio_values, color='green', label='{} Total Return: ${:.2f}'.format(agent.model_type, agent_return))
    ax[1].plot(dates, buy_and_hold_portfolio_values, color='blue', label='{} Buy and Hold Total Return: ${:.2f}'.format(stock_name, buy_and_hold_return))
    ax[1].set_ylabel('Portfolio Value ($)')
    ax[1].set_xticks(np.linspace(0, len(df), 10))
    ax[1].legend()
    ax[1].grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    plt.savefig('visualizations/{}_evaluation_{}.png'.format(model_name, stock_name,))


def plot_portfolio_returns_across_episodes(model_name, returns_across_episodes):
    len_episodes = len(returns_across_episodes)
    plt.figure(figsize=(15, 5), dpi=100)
    plt.title('Portfolio Returns')
    plt.plot(returns_across_episodes, color='black')
    plt.xlabel('Episode')
    plt.ylabel('Return Value')
    plt.grid()
    plt.savefig('visualizations/{}_returns_ep{}.png'.format(model_name, len_episodes))
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_action_distribution(action_distributions, model_name):
    action_array = np.array(action_distributions)
    plt.figure(figsize=(10, 6))
    plt.stackplot(
        range(len(action_array)),
        action_array[:, 0],
        action_array[:, 1],
        action_array[:, 2],
        labels=["Hold", "Buy", "Sell"]
    )
    plt.title(f"Action Distribution for Model: {model_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Action Proportion")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/{model_name}_action_distribution.png")
    plt.close()

def plot_portfolio_value(portfolio_values, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title(f'Portfolio Value Over Time ({model_name})')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig(f'plots/{model_name}_portfolio_value.png')
    plt.close()

def plot_daily_returns(daily_returns, model_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, kde=True, bins=30)
    plt.title(f'Daily Return Distribution ({model_name})')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.savefig(f'plots/{model_name}_daily_return_distribution.png')
    plt.close()

def plot_sharpe_ratios(sharpe_ratios, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(sharpe_ratios, label='Sharpe Ratio')
    plt.title(f'Sharpe Ratio Over Episodes ({model_name})')
    plt.xlabel('Episode')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.savefig(f'plots/{model_name}_sharpe_ratio.png')
    plt.close()

def plot_max_drawdowns(max_drawdowns, model_name):
    plt.figure(figsize=(10, 6))
    plt.boxplot(max_drawdowns)
    plt.title(f'Max Drawdowns ({model_name})')
    plt.ylabel('Max Drawdown')
    plt.savefig(f'plots/{model_name}_max_drawdowns.png')
    plt.close()

def plot_rewards(rewards_per_episode, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode, label='Cumulative Rewards')
    plt.title(f'Cumulative Rewards Over Episodes ({model_name})')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Rewards')
    plt.legend()
    plt.savefig(f'plots/{model_name}_cumulative_rewards.png')
    plt.close()

def plot_position_and_cash(portfolio_values, cash_balances, model_name):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(portfolio_values, label='Portfolio Value', color='blue')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Portfolio Value', color='blue')
    ax2 = ax1.twinx()
    ax2.plot(cash_balances, label='Cash Balance', color='green')
    ax2.set_ylabel('Cash Balance', color='green')
    plt.title(f'Portfolio Value and Cash Balance ({model_name})')
    fig.tight_layout()
    plt.savefig(f'plots/{model_name}_position_cash.png')
    plt.close()


def calculate_max_drawdown(portfolio_values):
    portfolio_values = np.array(portfolio_values)
    if len(portfolio_values) == 0:
        return 0.0
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)
    return max_drawdown


import numpy as np

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    daily_returns = np.array(daily_returns)
    if len(daily_returns) == 0:
        return 0.0
    mean_return = np.mean(daily_returns)
    volatility = np.std(daily_returns)
    if volatility == 0:
        return 0.0
    sharpe_ratio = (mean_return - risk_free_rate) / volatility
    return sharpe_ratio

def plot_evaluate_portfolio_value(portfolio_values, model_name,stock_name):
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title(f'Portfolio Value Over Time ({model_name}_{stock_name})')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.savefig(f'plots/evaluate_{model_name}_{stock_name}_portfolio_value.png')
    plt.close()

def plot_evaluate_daily_returns(daily_returns, model_name,stock_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, kde=True, bins=30)
    plt.title(f'Daily Return Distribution ({model_name}_{stock_name})')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.savefig(f'plots/evaluate_{model_name}_{stock_name}_daily_return_distribution.png')
    plt.close()
