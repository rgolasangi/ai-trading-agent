"""
Reinforcement Learning Trading Agent for AI Trading Agent
Implements Deep Q-Network (DQN) and Policy Gradient methods for options trading
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import pickle
import os
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning
    """
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000,
                 max_position_size: float = 0.1, transaction_cost: float = 0.001):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        
        # State space: [price, volume, sentiment, news_score, greeks, portfolio_value, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        
        # Action space: [0: Hold, 1: Buy Call, 2: Buy Put, 3: Sell Call, 4: Sell Put, 5: Close Position]
        self.action_space = spaces.Discrete(6)
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = []  # List of open positions
        self.trade_history = []
        self.max_portfolio_value = self.initial_balance
        self.drawdown = 0.0
        
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        
        # Execute action
        reward = self._execute_action(action, current_data)
        
        # Update portfolio value
        self._update_portfolio_value(current_data)
        
        # Calculate drawdown
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        self.drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.portfolio_value <= self.initial_balance * 0.5 or  # 50% loss
                self.drawdown > 0.3)  # 30% drawdown
        
        truncated = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, truncated, self._get_info()
    
    def _get_observation(self):
        """Get current state observation"""
        if self.current_step >= len(self.data):
            return np.zeros(20, dtype=np.float32)
        
        current_data = self.data.iloc[self.current_step]
        
        # Market features
        price = current_data.get('underlying_price', 0)
        volume = current_data.get('volume', 0)
        volatility = current_data.get('implied_volatility', 0.2)
        
        # Sentiment and news features
        sentiment_score = current_data.get('sentiment_score', 0)
        news_score = current_data.get('news_score', 0)
        
        # Greeks features
        delta = current_data.get('delta', 0)
        gamma = current_data.get('gamma', 0)
        theta = current_data.get('theta', 0)
        vega = current_data.get('vega', 0)
        
        # Technical indicators
        rsi = current_data.get('rsi', 50)
        macd = current_data.get('macd', 0)
        bb_position = current_data.get('bb_position', 0.5)
        
        # Portfolio features
        portfolio_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        position_count = len(self.positions)
        total_exposure = sum(pos['exposure'] for pos in self.positions)
        
        # Risk features
        var_95 = current_data.get('var_95', 0)
        sharpe_ratio = current_data.get('sharpe_ratio', 0)
        
        # Time features
        time_of_day = (self.current_step % 375) / 375  # Assuming 375 minutes in trading day
        day_of_week = current_data.get('day_of_week', 0) / 7
        
        observation = np.array([
            price / 20000,  # Normalize price
            volume / 1000000,  # Normalize volume
            volatility,
            sentiment_score,
            news_score,
            delta,
            gamma * 1000,  # Scale gamma
            theta / 100,  # Scale theta
            vega / 100,  # Scale vega
            rsi / 100,
            macd,
            bb_position,
            portfolio_return,
            position_count / 10,  # Normalize position count
            total_exposure / self.initial_balance,
            var_95,
            sharpe_ratio,
            self.drawdown,
            time_of_day,
            day_of_week
        ], dtype=np.float32)
        
        return observation
    
    def _execute_action(self, action, current_data):
        """Execute trading action and return reward"""
        reward = 0
        
        price = current_data.get('underlying_price', 0)
        option_price = current_data.get('option_price', 0)
        
        if action == 0:  # Hold
            reward = 0
        
        elif action == 1:  # Buy Call
            if self.balance > option_price * 100:  # Assuming lot size of 100
                position = {
                    'type': 'long_call',
                    'entry_price': option_price,
                    'quantity': 1,
                    'entry_time': self.current_step,
                    'exposure': option_price * 100
                }
                self.positions.append(position)
                self.balance -= option_price * 100 * (1 + self.transaction_cost)
                reward = 0.1  # Small positive reward for taking action
        
        elif action == 2:  # Buy Put
            if self.balance > option_price * 100:
                position = {
                    'type': 'long_put',
                    'entry_price': option_price,
                    'quantity': 1,
                    'entry_time': self.current_step,
                    'exposure': option_price * 100
                }
                self.positions.append(position)
                self.balance -= option_price * 100 * (1 + self.transaction_cost)
                reward = 0.1
        
        elif action == 3:  # Sell Call (Short)
            position = {
                'type': 'short_call',
                'entry_price': option_price,
                'quantity': 1,
                'entry_time': self.current_step,
                'exposure': option_price * 100
            }
            self.positions.append(position)
            self.balance += option_price * 100 * (1 - self.transaction_cost)
            reward = 0.1
        
        elif action == 4:  # Sell Put (Short)
            position = {
                'type': 'short_put',
                'entry_price': option_price,
                'quantity': 1,
                'entry_time': self.current_step,
                'exposure': option_price * 100
            }
            self.positions.append(position)
            self.balance += option_price * 100 * (1 - self.transaction_cost)
            reward = 0.1
        
        elif action == 5:  # Close Position
            if self.positions:
                position = self.positions.pop(0)  # Close oldest position
                pnl = self._calculate_position_pnl(position, current_data)
                self.balance += pnl
                reward = pnl / 1000  # Scale reward
                
                # Record trade
                self.trade_history.append({
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': option_price,
                    'pnl': pnl,
                    'holding_period': self.current_step - position['entry_time']
                })
        
        return reward
    
    def _calculate_position_pnl(self, position, current_data):
        """Calculate P&L for a position"""
        current_price = current_data.get('option_price', 0)
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        if position['type'] in ['long_call', 'long_put']:
            pnl = (current_price - entry_price) * quantity * 100
        else:  # short positions
            pnl = (entry_price - current_price) * quantity * 100
        
        # Apply transaction costs
        pnl -= abs(pnl) * self.transaction_cost
        
        return pnl
    
    def _update_portfolio_value(self, current_data):
        """Update total portfolio value"""
        total_position_value = 0
        
        for position in self.positions:
            position_value = self._calculate_position_value(position, current_data)
            total_position_value += position_value
        
        self.portfolio_value = self.balance + total_position_value
    
    def _calculate_position_value(self, position, current_data):
        """Calculate current value of a position"""
        current_price = current_data.get('option_price', 0)
        entry_price = position['entry_price']
        quantity = position['quantity']
        
        if position['type'] in ['long_call', 'long_put']:
            return current_price * quantity * 100
        else:  # short positions
            return -current_price * quantity * 100
    
    def _get_info(self):
        """Get additional info"""
        return {
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': len(self.positions),
            'drawdown': self.drawdown,
            'total_trades': len(self.trade_history)
        }

class DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int):
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000, batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Update target network
        self.update_target_network()
        
        self.logger = get_logger(__name__)
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

class PolicyGradientNetwork(nn.Module):
    """Policy Gradient Network for continuous action spaces"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(PolicyGradientNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Policy head (action probabilities)
        self.policy_head = nn.Linear(hidden_size // 2, action_size)
        
        # Value head (state value)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        policy = F.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x)
        
        return policy, value

class PolicyGradientAgent:
    """Policy Gradient Agent (Actor-Critic)"""
    
    def __init__(self, state_size: int, action_size: int, lr: float = 0.001,
                 gamma: float = 0.95):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        # Neural network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyGradientNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        
        self.logger = get_logger(__name__)
    
    def act(self, state):
        """Choose action using policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, value = self.network(state_tensor)
        
        # Sample action from policy
        action_dist = torch.distributions.Categorical(policy)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob, value
    
    def remember(self, state, action, reward, log_prob, value):
        """Store experience"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def update(self):
        """Update policy using collected experiences"""
        if len(self.rewards) == 0:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        # Normalize rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # Calculate advantages
        values = torch.cat(self.values).squeeze()
        advantages = discounted_rewards - values
        
        # Calculate losses
        policy_loss = []
        value_loss = []
        
        for log_prob, advantage, value, reward in zip(self.log_probs, advantages, values, discounted_rewards):
            policy_loss.append(-log_prob * advantage.detach())
            value_loss.append(F.mse_loss(value, reward.unsqueeze(0)))
        
        total_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        self.clear_memory()
    
    def clear_memory(self):
        """Clear episode memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class RLTradingAgent:
    """Main Reinforcement Learning Trading Agent"""
    
    def __init__(self, agent_type: str = 'dqn'):
        self.agent_type = agent_type
        self.logger = get_logger(__name__)
        
        # Initialize agent based on type
        if agent_type == 'dqn':
            self.agent = DQNAgent(state_size=20, action_size=6)
        elif agent_type == 'policy_gradient':
            self.agent = PolicyGradientAgent(state_size=20, action_size=6)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        self.training_history = []
        self.is_trained = False
    
    def train(self, training_data: pd.DataFrame, episodes: int = 1000,
              target_update_freq: int = 100, save_freq: int = 100):
        """
        Train the RL agent
        
        Args:
            training_data: Historical market data for training
            episodes: Number of training episodes
            target_update_freq: Frequency to update target network (DQN only)
            save_freq: Frequency to save model
        """
        try:
            self.logger.info(f"Starting RL agent training with {episodes} episodes...")
            
            env = TradingEnvironment(training_data)
            
            for episode in range(episodes):
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                
                while True:
                    if self.agent_type == 'dqn':
                        action = self.agent.act(state, training=True)
                        next_state, reward, done, truncated, info = env.step(action)
                        
                        self.agent.remember(state, action, reward, next_state, done or truncated)
                        self.agent.replay()
                        
                    elif self.agent_type == 'policy_gradient':
                        action, log_prob, value = self.agent.act(state)
                        next_state, reward, done, truncated, info = env.step(action)
                        
                        self.agent.remember(state, action, reward, log_prob, value)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    if done or truncated:
                        break
                
                # Update policy gradient agent at end of episode
                if self.agent_type == 'policy_gradient':
                    self.agent.update()
                
                # Update target network for DQN
                if self.agent_type == 'dqn' and episode % target_update_freq == 0:
                    self.agent.update_target_network()
                
                # Log progress
                if episode % 100 == 0:
                    portfolio_value = info.get('portfolio_value', 0)
                    self.logger.info(
                        f"Episode {episode}: Total Reward: {total_reward:.2f}, "
                        f"Portfolio Value: {portfolio_value:.2f}, Steps: {steps}"
                    )
                
                # Save model
                if episode % save_freq == 0 and episode > 0:
                    self.save_model(f"models/rl_agent_{self.agent_type}_episode_{episode}.pth")
                
                # Store training history
                self.training_history.append({
                    'episode': episode,
                    'total_reward': total_reward,
                    'portfolio_value': info.get('portfolio_value', 0),
                    'steps': steps,
                    'epsilon': getattr(self.agent, 'epsilon', 0)
                })
            
            self.is_trained = True
            self.logger.info("RL agent training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during RL agent training: {e}")
    
    def predict(self, state: np.ndarray) -> int:
        """
        Predict action for given state
        
        Args:
            state: Current market state
            
        Returns:
            Predicted action
        """
        try:
            if not self.is_trained:
                self.logger.warning("Agent not trained yet, returning random action")
                return np.random.randint(0, 6)
            
            if self.agent_type == 'dqn':
                return self.agent.act(state, training=False)
            elif self.agent_type == 'policy_gradient':
                action, _, _ = self.agent.act(state)
                return action
            
        except Exception as e:
            self.logger.error(f"Error predicting action: {e}")
            return 0  # Default to hold
    
    def get_trading_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading signals based on current market data
        
        Args:
            market_data: Current market data
            
        Returns:
            Trading signals
        """
        try:
            # Convert market data to state vector
            state = self._market_data_to_state(market_data)
            
            # Get action prediction
            action = self.predict(state)
            
            # Convert action to trading signal
            action_map = {
                0: 'HOLD',
                1: 'BUY_CALL',
                2: 'BUY_PUT',
                3: 'SELL_CALL',
                4: 'SELL_PUT',
                5: 'CLOSE_POSITION'
            }
            
            signal = action_map.get(action, 'HOLD')
            
            # Calculate confidence based on Q-values (for DQN)
            confidence = 0.5  # Default confidence
            if self.agent_type == 'dqn' and self.is_trained:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                q_values = self.agent.q_network(state_tensor)
                q_values_np = q_values.detach().cpu().numpy()[0]
                
                # Confidence based on difference between best and second-best action
                sorted_q = np.sort(q_values_np)[::-1]
                if len(sorted_q) > 1:
                    confidence = min((sorted_q[0] - sorted_q[1]) / abs(sorted_q[0]) + 0.5, 1.0)
            
            return {
                'signal': signal,
                'action_id': action,
                'confidence': confidence,
                'agent_type': self.agent_type,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return {
                'signal': 'HOLD',
                'action_id': 0,
                'confidence': 0.0,
                'agent_type': self.agent_type,
                'is_trained': self.is_trained,
                'timestamp': datetime.now()
            }
    
    def _market_data_to_state(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Convert market data dictionary to state vector"""
        try:
            # Extract features from market data
            price = market_data.get('underlying_price', 0) / 20000
            volume = market_data.get('volume', 0) / 1000000
            volatility = market_data.get('implied_volatility', 0.2)
            sentiment_score = market_data.get('sentiment_score', 0)
            news_score = market_data.get('news_score', 0)
            delta = market_data.get('delta', 0)
            gamma = market_data.get('gamma', 0) * 1000
            theta = market_data.get('theta', 0) / 100
            vega = market_data.get('vega', 0) / 100
            rsi = market_data.get('rsi', 50) / 100
            macd = market_data.get('macd', 0)
            bb_position = market_data.get('bb_position', 0.5)
            
            # Portfolio features (would come from portfolio manager)
            portfolio_return = market_data.get('portfolio_return', 0)
            position_count = market_data.get('position_count', 0) / 10
            total_exposure = market_data.get('total_exposure', 0)
            var_95 = market_data.get('var_95', 0)
            sharpe_ratio = market_data.get('sharpe_ratio', 0)
            drawdown = market_data.get('drawdown', 0)
            
            # Time features
            time_of_day = market_data.get('time_of_day', 0.5)
            day_of_week = market_data.get('day_of_week', 0) / 7
            
            state = np.array([
                price, volume, volatility, sentiment_score, news_score,
                delta, gamma, theta, vega, rsi, macd, bb_position,
                portfolio_return, position_count, total_exposure,
                var_95, sharpe_ratio, drawdown, time_of_day, day_of_week
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error converting market data to state: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.agent.save(filepath)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            self.agent.load(filepath)
            self.is_trained = True
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training performance metrics"""
        if not self.training_history:
            return {}
        
        history_df = pd.DataFrame(self.training_history)
        
        return {
            'total_episodes': len(self.training_history),
            'final_portfolio_value': history_df['portfolio_value'].iloc[-1],
            'max_portfolio_value': history_df['portfolio_value'].max(),
            'avg_reward': history_df['total_reward'].mean(),
            'reward_std': history_df['total_reward'].std(),
            'final_epsilon': history_df['epsilon'].iloc[-1] if 'epsilon' in history_df.columns else 0,
            'avg_steps_per_episode': history_df['steps'].mean()
        }

if __name__ == "__main__":
    # Test the RL agent
    import asyncio
    
    async def test_rl_agent():
        # Create sample training data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='15min')
        
        training_data = pd.DataFrame({
            'timestamp': dates,
            'underlying_price': 19500 + np.cumsum(np.random.randn(len(dates)) * 10),
            'option_price': 100 + np.random.randn(len(dates)) * 20,
            'volume': np.random.randint(1000, 10000, len(dates)),
            'implied_volatility': 0.2 + np.random.randn(len(dates)) * 0.05,
            'sentiment_score': np.random.randn(len(dates)) * 0.5,
            'news_score': np.random.randn(len(dates)) * 0.3,
            'delta': np.random.randn(len(dates)) * 0.5,
            'gamma': np.random.randn(len(dates)) * 0.01,
            'theta': np.random.randn(len(dates)) * 10,
            'vega': np.random.randn(len(dates)) * 50,
            'rsi': 30 + np.random.randn(len(dates)) * 20,
            'macd': np.random.randn(len(dates)) * 5,
            'bb_position': np.random.rand(len(dates))
        })
        
        # Test DQN agent
        print("Testing DQN Agent...")
        dqn_agent = RLTradingAgent(agent_type='dqn')
        
        # Train for a few episodes (reduced for testing)
        dqn_agent.train(training_data.head(1000), episodes=10)
        
        # Test prediction
        sample_data = {
            'underlying_price': 19550,
            'volume': 5000,
            'implied_volatility': 0.25,
            'sentiment_score': 0.1,
            'news_score': -0.05,
            'delta': 0.5,
            'gamma': 0.01,
            'theta': -5,
            'vega': 30,
            'rsi': 60,
            'macd': 2,
            'bb_position': 0.7
        }
        
        signals = dqn_agent.get_trading_signals(sample_data)
        print(f"DQN Trading Signal: {signals}")
        
        # Get training metrics
        metrics = dqn_agent.get_training_metrics()
        print(f"DQN Training Metrics: {metrics}")
        
        # Test Policy Gradient agent
        print("\nTesting Policy Gradient Agent...")
        pg_agent = RLTradingAgent(agent_type='policy_gradient')
        
        # Train for a few episodes
        pg_agent.train(training_data.head(1000), episodes=10)
        
        # Test prediction
        signals = pg_agent.get_trading_signals(sample_data)
        print(f"Policy Gradient Trading Signal: {signals}")
        
        # Get training metrics
        metrics = pg_agent.get_training_metrics()
        print(f"Policy Gradient Training Metrics: {metrics}")
    
    asyncio.run(test_rl_agent())

