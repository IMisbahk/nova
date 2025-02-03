import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, dataPath, initialBalance=10000, windowSize=50):
        super(TradingEnv, self).__init__()
        self.data = pd.read_csv(dataPath)
        self.initialBalance = initialBalance
        self.windowSize = windowSize
        self.actionSpace = spaces.Discrete(3)
        self.observationSpace = spaces.Box(low=-np.inf, high=np.inf, shape=(windowSize + 2,), dtype=np.float32)
        self.reset()
    
    def reset(self):
        self.currentStep = self.windowSize
        self.balance = self.initialBalance
        self.sharesHeld = 0
        self.netWorth = self.initialBalance
        return self._nextObservation()
    
    def _nextObservation(self):
        obs = np.concatenate([
            self.data.iloc[self.currentStep - self.windowSize:self.currentStep]['Close'].values,
            [self.balance],
            [self.sharesHeld]
        ])
        return obs
    
    def step(self, action):
        currentPrice = self.data.iloc[self.currentStep]['Close']
        if action == 1:
            numShares = self.balance // currentPrice
            self.sharesHeld += numShares
            self.balance -= numShares * currentPrice
        elif action == 2:
            self.balance += self.sharesHeld * currentPrice
            self.sharesHeld = 0
        
        self.currentStep += 1
        self.netWorth = self.balance + (self.sharesHeld * currentPrice)
        done = self.currentStep >= len(self.data) - 1 or self.netWorth <= 0
        reward = self.netWorth - self.initialBalance
        
        return self._nextObservation(), reward, done, {}
