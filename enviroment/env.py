import numpy as np
import pandas as pd
from datetime import date, timedelta

class StockMarket:
    def __init__(self, max_episode_steps=300, epsilon=0.05):
        self.params = 537
        
        self.sectors = ['XLK', 'XLF', 'XLY', 'XLI', 'XLB', 'XLV', 'XLU', 'XLP', 'XLE', 'XLRE', 'XLC']
        self.data = pd.read_csv(f'enviroment/norm-{self.sectors[0]}.csv', index_col=0)
        for sector in self.sectors[1:]:
            sector_data = pd.read_csv(f'enviroment/norm-{sector}.csv', index_col=0)
            self.data = self.data.join(sector_data, how='outer', rsuffix=f'_{sector}')
        self.data.fillna(0, inplace=True)
        self.data.rename(columns={f'open': f'open_{self.sectors[0]}'}, inplace=True)
        
        self.spy_data = pd.read_csv(f'enviroment/norm-SPY.csv', index_col=0)
        self.spy_data = self.spy_data[['open', 'high', 'low', 'close', 'volume']]
        self.spy_data.fillna(0, inplace=True)
        
        self.price_cal = self.data.copy()
        self.price_cal = self.price_cal[[f'open_{sector}' for sector in self.sectors]]
        self.price_cal = self.price_cal.map(lambda x: x + 1)
        
        self.cash = 10_000
        self.date = date(2000, 1, 3)
        self.final_date = date(2024, 11, 29)
        
        self.max_episode_steps = max_episode_steps
        self.steps = 0
        
        self.position_text = ['CASH', 'XLK', 'XLF', 'XLY', 'XLI', 'XLB', 'XLV', 'XLU', 'XLP', 'XLE', 'XLRE', 'XLC']
        self.previos_position = np.zeros(12)
        self.previos_position[0] = 1
        
        self.epsilon = epsilon
        
    def reset(self, start_date):
        self.cash = 10_000
        self.date = start_date
        self.steps = 0
            
    def get_data_for_date(self, date):
        str_date = date.strftime('%Y-%m-%d')
        final_data = self.data[self.data.index == str_date].values
        
        if len(final_data) != 1:
            return None
            
        return final_data.flatten()
    
    def get_spy_data_for_date(self, date):
        str_date = date.strftime('%Y-%m-%d')
        final_data = self.spy_data[self.spy_data.index == str_date].values
        
        if len(final_data) != 1:
            return None
        
        return final_data.flatten()
        
    def get_curr_state(self):
        date_data = self.get_data_for_date(self.date)
        if date_data is None:
            return None, None
        
        return date_data, self.previos_position
    
    def get_curr_spy_state(self):
        date_data = self.get_spy_data_for_date(self.date)
        if date_data is None:
            return None
        
        return date_data
    
    def step(self, action):
        if self.steps >= self.max_episode_steps or self.date >= self.final_date:
            return None, None, None, None
        
        self.steps += 1
        
        date_data, _ = self.get_curr_state()
        if date_data is None:
            return None, None, None, None
        
        reward = 0
        done = False
        
        actual_action = action.copy()
        
        changes = np.ones(len(action))
        changes[1:] = self.price_cal[self.price_cal.index == self.date.strftime('%Y-%m-%d')].values.flatten()
        
        random_change = (np.random.dirichlet(np.ones(len(action)), 1) * self.epsilon)[0]
        action += random_change - (action * self.epsilon)
        action = action / np.sum(action)
        
        old_date = self.date
        self.date += timedelta(days=1)
        while self.date.weekday() >= 5:
            self.date += timedelta(days=1)
        while self.date.strftime('%Y-%m-%d') not in self.data.index:
            self.date += timedelta(days=1)
        
        changes = np.ones(len(action))
        changes[1:] = self.price_cal[self.price_cal.index == self.date.strftime('%Y-%m-%d')].values.flatten()
        
        new_cash = np.sum(self.cash * action * changes)
        reward = (new_cash - self.cash)
        self.cash = new_cash
        self.previos_position = action * changes
        self.previos_position = self.previos_position / np.sum(self.previos_position)
            
        if self.date >= self.final_date or self.steps >= self.max_episode_steps or self.cash < 0:
            done = True
           
        return old_date, actual_action, reward, done, self.date, self.previos_position