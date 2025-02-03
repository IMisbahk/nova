from env import TradingEnv
from agent import DQNAgent
import torch

def train():
    env = TradingEnv('data/prices.csv')
    state_size = env.observationSpace.shape[0]
    action_size = env.actionSpace.n
    agent = DQNAgent(state_size, action_size)
    
    episodes = 1000
    batch_size = 32
    
    for e in range(episodes):
        state = env.reset()
        totalreward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            totalreward += reward
            
            if done:
                print(f"Episode: {e+1}/{episodes}, Total Reward: {totalreward:.2f}")
                break
            
            agent.replay(batch_size)
        
        #saving model regularly
        if e % 50 == 0:
            torch.save(agent.model.state_dict(), f'models/nova_model_{e}.pth')

if __name__ == '__main__':
    train()