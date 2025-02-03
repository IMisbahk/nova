# Nova - Reinforcement Learning Trading Agent

## Overview
Nova is a reinforcement learning-based trading agent that predicts future prices, simulates trades, and learns from its mistakes to improve over time. The goal is to create an AI trader that makes profitable decisions in financial markets.

## Features
- Uses reinforcement learning to optimize trading strategies.
- Simulates trades based on historical price data.
- Learns from bad trades and adapts over time.
- Simple UI for interacting with the agent.

## Folder Structure
```
Nova/
├── README.md             # Project overview and setup
├── requirements.txt      # Dependencies
├── data/                 # Store historical price data
│   └── prices.csv
├── src/                  # Main source code
│   ├── agent.py          # RL agent logic
│   ├── train.py          # Training script
│   ├── env.py            # Trading environment
│   ├── ui.py             # Simple UI to interact with Nova
└── models/               # Store trained models
    └── nova_model.zip
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Nova.git
   cd Nova
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare data:
   - Place historical price data in `data/prices.csv`

## Usage
### Train the Agent
Run the training script to start training Nova:
```bash
python src/train.py
```

### Run the UI
To interact with Nova through the UI:
```bash
python src/ui.py
```

## Future Plans
- Implement advanced technical indicators.
- Integrate real-time market data.
- Deploy as a web-based trading assistant.

## License
MIT License

