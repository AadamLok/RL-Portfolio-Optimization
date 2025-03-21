from enviroment.env import StockMarket
from datetime import date
import numpy as np

stock_market = StockMarket(max_episode_steps=10000)
stock_market.reset(date(2020, 1, 2))
for i in range(2000):
    # changes = np.ones(12)
    # changes[1:] = stock_market.price_cal[stock_market.price_cal.index == stock_market.date.strftime('%Y-%m-%d')].values.flatten()
    # max_index = np.argmax(changes)
    # action = np.full(12, 0.0)
    # action[max_index] = 1.0
    # stock_market.step(action)
    action = np.full(12, 1/11)
    action[0] = 0
    stock_market.step(action)
print(stock_market.date)
print(stock_market.cash)