from enviroment.env import StockMarket
from model.model import SectorModel, FullMarketModel, ActorCriticModel
from datetime import date

import numpy as np

import torch

def train_sector_model(sector):
    print(f'Training model for sector: {sector}')
    
    sm = StockMarket()
    model = SectorModel()
    model.initialize_weights()
    
    sector_index = sm.sectors.index(sector)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    
    for epoch in range(100):
        sm.reset(date(2000, 1, 3))
        curr_state = sm.get_curr_state()
        h0 = torch.zeros(2, 150)
        c0 = torch.zeros(2, 150)
        
        done = False
        
        total_loss = 0
        while not done:
            print(f'{sm.date}', end='\r')
            optimizer.zero_grad()
            
            curr_state = torch.tensor(sm.get_curr_state()[sector_index*537:(sector_index+1)*537]).float().view(1, -1)
            ohlcv, h0, c0 = model(torch.tensor(curr_state).float().view(1, -1), h0, c0)
            _,_,_,done,_ = sm.step(np.full(12, 1/12))
            
            next_state = torch.tensor(sm.get_curr_state()[sector_index*537:sector_index*537+5]).float().view(1, -1)
            
            loss = loss_fn(ohlcv, next_state)
            loss.backward(retain_graph=True)
            total_loss += loss.item()
            optimizer.step()
        
        print(f'Epoch: {epoch}, Loss: {total_loss}')
        total_loss = 0
        
def test_market_model():
    print(f'Training model for Full Market')
    
    sm = StockMarket()
    model = FullMarketModel().cuda()
    model.initialize_weights()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(100):
        sm.reset(date(2000, 1, 3))
        curr_state = sm.get_curr_state()
        h0 = torch.zeros(2, 150).cuda()
        c0 = torch.zeros(2, 150).cuda()
        hidden_sector = [(torch.zeros(2, 150).cuda(), torch.zeros(2, 150).cuda()) for _ in range(11)]
        
        done = False
        
        total_market_loss = 0
        total_sector_loss = [0 for _ in range(11)]
        loss_few_run = 0
        runs = 0
        while not done:
            print(f'{sm.date}', end='\r')
                
            curr_state = torch.tensor(sm.get_curr_state()).float().view(1, -1).cuda()
            ohlcv, h0, c0, hidden_sector, sector_out = model(curr_state, h0, c0, hidden_sector)
            _,_,_,done,_ = sm.step(np.full(12, 1/12))
            
            sectors_ohlcv = torch.tensor(sm.get_curr_state()).float().view(1, -1).cuda()
            sectors_ohlcv = [sectors_ohlcv[:, i*537:i*537+5] for i in range(11)]
            spy_ohlcv = torch.tensor(sm.get_curr_spy_state()).float().view(1, -1).cuda()
            
            sector_loss = [loss_fn(out, sectors_ohlcv[i]) for i, out in enumerate(sector_out)]
            market_loss = loss_fn(ohlcv, spy_ohlcv)
            loss_few_run += sum(sector_loss) + market_loss
            total_market_loss += market_loss.item()
            total_sector_loss = [total_sector_loss[i] + sector_loss[i].item() for i in range(11)]

            runs += 1
            if runs % 10 == 0:
                optimizer.zero_grad()   
                loss_few_run.backward()
                optimizer.step()
                runs = 0
                loss_few_run = 0
                h0 = h0.detach()
                c0 = c0.detach()
                hidden_sector = [(h.detach(), c.detach()) for h, c in hidden_sector]
        if runs != 0:
            optimizer.zero_grad()   
            loss_few_run.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch}, Loss: {total_market_loss}, Sector Loss: {total_sector_loss}')
        total_sector_loss = [0 for _ in range(11)]
        total_market_loss = 0

def test_actor_critic_model():
    print(f'Training actor critic model')
    
    sm = StockMarket()
    model = ActorCriticModel().cuda()
    model.initialize_weights()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(100):
        sm.reset(date(2000, 1, 3))
        curr_state = sm.get_curr_state()
        h0 = torch.zeros(2, 50).cuda()
        c0 = torch.zeros(2, 50).cuda()
        market_h0 = torch.zeros(2, 150).cuda()
        market_c0 = torch.zeros(2, 150).cuda()
        hidden_sector = [(torch.zeros(2, 150).cuda(), torch.zeros(2, 150).cuda()) for _ in range(11)]
        prev_hidden = ((market_h0, market_c0), hidden_sector)
        
        
        done = False
        
        total_market_loss = 0
        total_sector_loss = [0 for _ in range(11)]
        loss_few_run = 0
        runs = 0
        while not done:
            print(f'{sm.date}', end='\r')
                
            curr_state = torch.tensor(sm.get_curr_state()).float().view(1, -1).cuda()
            ohlcv, h0, c0, hidden_sector, sector_out = model(curr_state, h0, c0, hidden_sector)
            _,_,_,done,_ = sm.step(np.full(12, 1/12))
            
            sectors_ohlcv = torch.tensor(sm.get_curr_state()).float().view(1, -1).cuda()
            sectors_ohlcv = [sectors_ohlcv[:, i*537:i*537+5] for i in range(11)]
            spy_ohlcv = torch.tensor(sm.get_curr_spy_state()).float().view(1, -1).cuda()
            
            sector_loss = [loss_fn(out, sectors_ohlcv[i]) for i, out in enumerate(sector_out)]
            market_loss = loss_fn(ohlcv, spy_ohlcv)
            loss_few_run += sum(sector_loss) + market_loss
            total_market_loss += market_loss.item()
            total_sector_loss = [total_sector_loss[i] + sector_loss[i].item() for i in range(11)]

            runs += 1
            if runs % 10 == 0:
                optimizer.zero_grad()   
                loss_few_run.backward()
                optimizer.step()
                runs = 0
                loss_few_run = 0
                h0 = h0.detach()
                c0 = c0.detach()
                hidden_sector = [(h.detach(), c.detach()) for h, c in hidden_sector]
        if runs != 0:
            optimizer.zero_grad()   
            loss_few_run.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch}, Loss: {total_market_loss}, Sector Loss: {total_sector_loss}')
        total_sector_loss = [0 for _ in range(11)]
        total_market_loss = 0

if __name__ == '__main__':
    test_market_model()