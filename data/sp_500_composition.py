import pandas as pd
from datetime import datetime

class SP500Composition:
    def __init__(self, start_date=datetime(2000, 1, 1)):
        self.begnning_data = pd.read_csv('data/sp_500_composition.csv')
        
        additions = pd.read_csv('data/sp_500_additions.csv')
        additions['Action'] = 'Addition'
        removals = pd.read_csv('data/sp_500_removals.csv')
        removals['Action'] = 'Removal'
        self.all_changes = pd.concat([additions, removals])
        self.all_changes = self.all_changes.sort_values('Date')
        
        if start_date < datetime(2000, 1, 1):
            raise ValueError('Start date must be after 2000-01-01')
        self.curr_date = start_date
        self.curr_tickers = self.get_tickers_for(start_date)
        
    def get_tickers_for(self, date):
        start_tickers = set(self.begnning_data['Ticker'])
        
        for _, change in self.all_changes.iterrows():
            if change['Date'] > date:
                break
            if change['Action'] == 'Addition':
                start_tickers.add(change['Ticker'])
            else:
                start_tickers.remove(change['Ticker'])
        
        return start_tickers
    
    def change_to_next_date(self):
        self.curr_date = self.curr_date + pd.DateOffset(days=1)
        changes = self.all_changes[self.all_changes['Date'] == self.curr_date]
        
        for _, change in changes.iterrows():
            if change['Action'] == 'Addition':
                self.curr_tickers.add(change['Ticker'])
            else:
                self.curr_tickers.remove(change['Ticker'])
        
        return self.curr_tickers
        