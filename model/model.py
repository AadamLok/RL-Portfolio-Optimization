import torch
from torch import nn
from torch.nn import functional as F

class SectorModel(nn.Module):
    def __init__(self, num_inputs=537):
        super(SectorModel, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 300)
        self.fc2 = nn.Linear(300, 150)
        
        self.lstm = nn.LSTM(150, 150, 2, batch_first=True)
        
        self.fc3 = nn.Linear(150, 50)
        self.fc4 = nn.Linear(50, 5)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)
    
    def forward(self, x, h0, c0, middle=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x, (hn, cn) = self.lstm(x, (h0, c0))
        
        x = F.relu(self.fc3(x))
        out = self.fc4(x)
        
        if middle:
            return out, hn, cn, x
        
        return out, hn, cn

class FullMarketModel(nn.Module):
    def __init__(self, num_sectors=11, num_inputs=50):
        super(FullMarketModel, self).__init__()
        
        self.sector_models = nn.ModuleList([SectorModel().cuda() for _ in range(num_sectors)])
        
        self.fc1 = nn.Linear(num_inputs*num_sectors, 300)
        self.fc2 = nn.Linear(300, 150)
        
        self.lstm = nn.LSTM(150, 150, 2, batch_first=True)
        
        self.fc3 = nn.Linear(150, 50)
        self.fc4 = nn.Linear(50, 5)
        
    def initialize_weights(self):
        for sm in self.sector_models:
            sm.initialize_weights()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)

    def forward(self, x, h0, c0, sector_hidden, middle=False):
        sector_outs = [model(x[:, i*537:(i+1)*537], sector_hidden[i][0], sector_hidden[i][1], middle=True) for i, model in enumerate(self.sector_models)]
        
        sector_out = [sector_out for sector_out, _, _, _ in sector_outs]
        sector_hidden = [(hn, cn) for _, hn, cn, _ in sector_outs]
        x = torch.cat([x for _, _, _, x in sector_outs], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x, (hn, cn) = self.lstm(x, (h0, c0))
        
        out = F.relu(self.fc3(x))
        out = self.fc4(x)
        
        if middle:
            return out, hn, cn, sector_hidden, sector_out, x
        
        return out, hn, cn, sector_hidden, sector_out
    
class ActorCriticModel(nn.Module):
    def __init__(self, num_position=12, num_input=150):
        super(ActorCriticModel, self).__init__()
        self.full_market_model = FullMarketModel().cuda()
        self.fc1 = nn.Linear(num_input, 50)
        
        self.lstm = nn.LSTM(50+num_position, 50, 2, batch_first=True)
        
        self.fc2 = nn.Linear(50, 20)
        
        self.actor = nn.Linear(20, num_position)
        self.critic = nn.Linear(20, 1)
        
    def initialize_weights(self):
        self.full_market_model.initialize_weights()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    else:
                        nn.init.constant_(param, 0)
                        
    def forward(self, x, h0, c0, prev_hidden, position):
        market_out, market_h, market_c, sector_hidden, sector_out, x = self.full_market_model(x, prev_hidden[0][0], prev_hidden[0][1], prev_hidden[1], middle=True)
        prev_hidden = ((market_h, market_c), sector_hidden)
        
        x = F.relu(self.fc1(x))
        x = torch.cat([x, position], dim=1)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        x = F.relu(self.fc2(x))
        
        actor = F.softmax(self.actor(x), dim=1)
        critic = self.critic(x)
        
        return actor, critic, hn, cn, prev_hidden, market_out, sector_out
        