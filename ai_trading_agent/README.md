# AI Trading Agent for Nifty & Bank Nifty Options

## 🚀 Overview

A sophisticated AI-powered trading agent designed for automated options trading on Nifty and Bank Nifty indices. The system combines reinforcement learning, deep learning, and multi-agent architecture to achieve high-performance trading with comprehensive risk management.

## ✨ Key Features

### 🤖 Multi-Agent AI Architecture
- **Sentiment Analysis Agent**: Analyzes market sentiment from news and social media
- **News Analysis Agent**: Processes financial news and events for market impact
- **Options Greeks Agent**: Calculates and analyzes options Greeks for risk assessment
- **Reinforcement Learning Agent**: Makes intelligent trading decisions using deep RL

### 📊 Advanced Analytics
- **Real-time Market Data**: Live options and underlying data collection
- **Options Greeks Calculation**: Delta, Gamma, Theta, Vega, Rho analysis
- **Volatility Surface Analysis**: Implied volatility tracking and analysis
- **Portfolio Risk Metrics**: VaR, Sharpe ratio, drawdown analysis

### 🛡️ Comprehensive Risk Management
- **Position Sizing**: Automated position sizing based on Kelly criterion
- **Risk Limits**: Configurable exposure, drawdown, and concentration limits
- **Real-time Monitoring**: Continuous portfolio and risk monitoring
- **Emergency Controls**: Instant trading halt and emergency stop mechanisms

### 📈 Performance Optimization
- **Backtesting Engine**: Comprehensive strategy testing and validation
- **Performance Analytics**: Detailed performance metrics and reporting
- **Strategy Evaluation**: Multi-strategy comparison and optimization
- **Walk-forward Analysis**: Out-of-sample performance validation

### 🖥️ Professional Dashboard
- **Real-time Monitoring**: Live portfolio and position tracking
- **Interactive Charts**: Performance visualization and analytics
- **Risk Dashboard**: Real-time risk metrics and alerts
- **Trade Management**: Order tracking and execution monitoring

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI Trading Agent                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Data Layer    │  │   AI Agents     │  │  Execution   │ │
│  │                 │  │                 │  │              │ │
│  │ • Market Data   │  │ • Sentiment     │  │ • Order Mgmt │ │
│  │ • News Feed     │  │ • News Analysis │  │ • Portfolio  │ │
│  │ • Social Media  │  │ • Greeks Calc   │  │ • Risk Mgmt  │ │
│  │ • Options Data  │  │ • RL Trading    │  │ • Zerodha API│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Analytics     │  │   Monitoring    │  │  Dashboard   │ │
│  │                 │  │                 │  │              │ │
│  │ • Backtesting   │  │ • Performance   │  │ • React UI   │ │
│  │ • Performance   │  │ • Risk Alerts   │  │ • Real-time  │ │
│  │ • Optimization  │  │ • System Health │  │ • Charts     │ │
│  │ • Reporting     │  │ • Logging       │  │ • Controls   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Target Performance

- **Success Rate**: 90%+ target accuracy
- **Risk-Adjusted Returns**: Sharpe ratio > 2.0
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 70%
- **Profit Factor**: > 2.0

## 🛠️ Technology Stack

### Backend
- **Python 3.11**: Core application development
- **FastAPI**: High-performance API framework
- **PostgreSQL**: Primary database for structured data
- **Redis**: Caching and real-time data
- **MongoDB**: Document storage for unstructured data

### AI/ML
- **TensorFlow/PyTorch**: Deep learning frameworks
- **OpenAI GPT**: Natural language processing
- **scikit-learn**: Machine learning utilities
- **TA-Lib**: Technical analysis indicators
- **NumPy/Pandas**: Data processing and analysis

### Frontend
- **React 18**: Modern UI framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization
- **shadcn/ui**: Component library

### Infrastructure
- **Google Cloud Platform**: Cloud infrastructure
- **Docker**: Containerization
- **Nginx**: Web server and reverse proxy
- **Supervisor**: Process management

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 100GB SSD minimum
- **Network**: Stable internet connection with low latency

### API Requirements
- **Zerodha Kite Connect**: API key, secret, and access token
- **OpenAI API**: API key for GPT models
- **News API**: API key for news data (optional)
- **Social Media APIs**: Twitter/Reddit API access (optional)

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/your-username/ai-trading-agent.git
cd ai-trading-agent
```

### 2. Environment Setup
```bash
# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

Required environment variables:
```bash
# Zerodha API Configuration
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost/trading_db
REDIS_URL=redis://localhost:6379

# AI Configuration
OPENAI_API_KEY=your_openai_api_key

# Risk Management
MAX_POSITION_SIZE=0.1
MAX_DAILY_LOSS=0.05
RISK_TOLERANCE=medium
```

### 4. Database Setup
```bash
# Start PostgreSQL and Redis
sudo systemctl start postgresql redis

# Create database
createdb trading_db

# Run migrations
python manage.py migrate
```

### 5. Start Application
```bash
# Start main trading agent
python main.py

# In another terminal, start dashboard
cd trading-dashboard
npm install
npm run dev
```

### 6. Access Dashboard
Open your browser and navigate to `http://localhost:5173`

## 📖 Detailed Setup

### Database Configuration

#### PostgreSQL Setup
```sql
-- Create database and user
CREATE DATABASE trading_db;
CREATE USER trading_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_user;

-- Create tables (handled by migrations)
```

#### Redis Configuration
```bash
# Configure Redis for caching
sudo nano /etc/redis/redis.conf

# Set memory policy
maxmemory-policy allkeys-lru
maxmemory 1gb
```

### AI Model Setup

#### Download Pre-trained Models
```bash
# Create models directory
mkdir -p models/

# Download sentiment analysis model
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
tokenizer.save_pretrained('models/finbert')
model.save_pretrained('models/finbert')
"
```

#### Initialize RL Agent
```bash
# Train initial RL model (optional - pre-trained model included)
python agents/rl_agent.py --train --episodes 1000
```

### Risk Management Configuration

#### Set Risk Limits
```python
# config/risk_config.py
RISK_LIMITS = {
    'max_position_size': 0.1,  # 10% per position
    'max_portfolio_exposure': 0.8,  # 80% total exposure
    'max_daily_loss': 0.05,  # 5% daily loss limit
    'max_drawdown': 0.15,  # 15% maximum drawdown
    'var_limit': 0.03,  # 3% VaR limit
    'max_correlation': 0.7,  # Maximum correlation
    'min_liquidity_ratio': 0.2,  # 20% cash minimum
    'max_leverage': 3.0,  # 3x maximum leverage
}
```

## 🔧 Configuration

### Trading Configuration
```python
# config/trading_config.py
TRADING_CONFIG = {
    'instruments': ['NIFTY', 'BANKNIFTY'],
    'option_types': ['CE', 'PE'],
    'expiry_days': [0, 7, 14, 30],  # Days to expiry to trade
    'strike_range': 10,  # Number of strikes around ATM
    'min_volume': 1000,  # Minimum volume filter
    'max_spread': 0.05,  # Maximum bid-ask spread
    'trading_hours': {
        'start': '09:15',
        'end': '15:30'
    }
}
```

### AI Agent Configuration
```python
# config/agent_config.py
AGENT_CONFIG = {
    'sentiment_agent': {
        'model': 'finbert',
        'confidence_threshold': 0.7,
        'update_frequency': 300  # seconds
    },
    'news_agent': {
        'sources': ['reuters', 'bloomberg', 'economic_times'],
        'relevance_threshold': 0.8,
        'max_articles_per_hour': 50
    },
    'greeks_agent': {
        'calculation_method': 'black_scholes',
        'volatility_model': 'garch',
        'update_frequency': 60  # seconds
    },
    'rl_agent': {
        'model_type': 'dqn',
        'learning_rate': 0.001,
        'epsilon_decay': 0.995,
        'memory_size': 10000
    }
}
```

## 📊 Usage Examples

### Basic Trading
```python
from trading.execution_engine import ExecutionEngine
from agents.agent_coordinator import AgentCoordinator

# Initialize components
coordinator = AgentCoordinator()
execution_engine = ExecutionEngine()

# Generate trading signals
market_data = {
    'symbol': 'NIFTY24APR19500CE',
    'price': 150,
    'volume': 50000,
    'underlying_price': 19000
}

signals = await coordinator.generate_signals(market_data)

# Execute trades
for signal in signals:
    result = execution_engine.execute_trade(signal)
    print(f"Trade executed: {result}")
```

### Risk Management
```python
from risk_management.risk_manager import RiskManager

# Initialize risk manager
risk_manager = RiskManager(initial_capital=100000)

# Validate trade signal
is_valid, reason = risk_manager.validate_trade_signal(signal, current_positions)

if is_valid:
    # Execute trade
    execute_trade(signal)
else:
    print(f"Trade rejected: {reason}")
```

### Backtesting
```python
from backtesting.backtester import Backtester
from backtesting.strategy_evaluator import StrategyEvaluator

# Define strategy
async def my_strategy(data):
    # Your trading logic here
    if condition:
        return TradeSignal(...)
    return None

# Run backtest
backtester = Backtester(initial_capital=100000)
backtester.set_strategy(my_strategy)
result = await backtester.run_backtest(historical_data)

# Evaluate performance
evaluator = StrategyEvaluator()
analysis = evaluator.analyze_performance(result)
print(f"Sharpe Ratio: {analysis['sharpe_ratio']}")
```

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_agents.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Run Integration Tests
```bash
# Run integration test suite
python tests/test_integration.py

# Expected output:
# ✓ Sentiment Agent tests passed
# ✓ News Agent tests passed
# ✓ Greeks Agent tests passed
# ✓ RL Agent tests passed
# ✓ Risk Manager tests passed
# ✓ Execution Engine tests passed
# ✓ End-to-End Workflow tests passed
# SUCCESS RATE: 95.0%
```

### Performance Testing
```bash
# Test signal generation speed
python tests/test_performance.py

# Load testing
python tests/test_load.py --concurrent-users 10 --duration 60
```

## 📈 Monitoring

### System Health
```bash
# Check system status
python utils/health_check.py

# Monitor logs
tail -f logs/trading_agent.log

# Monitor performance
python utils/performance_monitor.py
```

### Dashboard Metrics
- **Portfolio Value**: Real-time portfolio valuation
- **Daily P&L**: Daily profit and loss tracking
- **Risk Metrics**: VaR, drawdown, exposure monitoring
- **AI Agent Status**: Agent health and confidence levels
- **Trade Performance**: Win rate, profit factor, Sharpe ratio

## 🚨 Risk Management

### Pre-Trade Checks
- Position size validation
- Portfolio exposure limits
- Correlation risk assessment
- Liquidity requirements
- Leverage constraints

### Real-Time Monitoring
- Continuous risk metric calculation
- Automated alert generation
- Emergency stop mechanisms
- Position limit enforcement

### Risk Alerts
- **Low Risk**: Informational alerts
- **Medium Risk**: Warning alerts
- **High Risk**: Action required alerts
- **Critical Risk**: Automatic trading halt

## 🔒 Security

### API Security
- Secure credential storage
- API rate limiting
- Request validation
- Error handling

### Data Security
- Encrypted data storage
- Secure data transmission
- Access control
- Audit logging

### System Security
- Regular security updates
- Firewall configuration
- Intrusion detection
- Backup and recovery

## 📚 Documentation

### API Documentation
- **Swagger UI**: Available at `/docs` when running
- **ReDoc**: Available at `/redoc`
- **Postman Collection**: `docs/api_collection.json`

### Code Documentation
- **Docstrings**: Comprehensive function documentation
- **Type Hints**: Full type annotation
- **Comments**: Inline code explanations

### User Guides
- **Trading Guide**: `docs/trading_guide.md`
- **Risk Management**: `docs/risk_management.md`
- **Troubleshooting**: `docs/troubleshooting.md`

## 🤝 Contributing

### Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/ai-trading-agent.git

# Create feature branch
git checkout -b feature/new-feature

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### Code Standards
- **PEP 8**: Python code style
- **Type Hints**: Full type annotation
- **Docstrings**: Google style docstrings
- **Testing**: Minimum 80% code coverage

### Pull Request Process
1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request
5. Code review and approval

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading in financial markets involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. The authors and contributors are not responsible for any financial losses incurred through the use of this software.

## 📞 Support

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Community discussions and Q&A
- **Wiki**: Additional documentation and guides

### Professional Support
- **Email**: support@yourcompany.com
- **Documentation**: https://docs.yourcompany.com
- **Training**: Professional training available

### Emergency Contact
- **Critical Issues**: emergency@yourcompany.com
- **Phone**: +1-XXX-XXX-XXXX (24/7 for production issues)

---

## 🎉 Acknowledgments

- **Zerodha**: For providing excellent API infrastructure
- **OpenAI**: For advanced language models
- **Open Source Community**: For the amazing libraries and tools
- **Contributors**: All the developers who made this project possible

---

**Happy Trading! 🚀📈**

