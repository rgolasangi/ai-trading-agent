import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { 
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts'
import { 
  TrendingUp, TrendingDown, DollarSign, Activity, AlertTriangle, 
  Target, Brain, BarChart3, Settings, Play, Pause, Square,
  Eye, Shield, Zap, Users, Clock, CheckCircle, XCircle
} from 'lucide-react'
import './App.css'

// Mock data for demonstration
const mockPortfolioData = {
  totalValue: 125000,
  dailyPnL: 2500,
  totalPnL: 25000,
  dailyReturn: 2.04,
  totalReturn: 25.0,
  positions: 8,
  cash: 45000,
  margin: 80000
}

const mockPerformanceData = [
  { date: '2024-01-01', value: 100000, benchmark: 100000 },
  { date: '2024-01-15', value: 102000, benchmark: 101000 },
  { date: '2024-02-01', value: 105000, benchmark: 102500 },
  { date: '2024-02-15', value: 108000, benchmark: 103000 },
  { date: '2024-03-01', value: 112000, benchmark: 104000 },
  { date: '2024-03-15', value: 118000, benchmark: 105500 },
  { date: '2024-04-01', value: 125000, benchmark: 106000 }
]

const mockPositions = [
  { symbol: 'NIFTY24APR19500CE', quantity: 100, avgPrice: 150, ltp: 175, pnl: 2500, pnlPct: 16.67 },
  { symbol: 'BANKNIFTY24APR45000PE', quantity: -50, avgPrice: 200, ltp: 180, pnl: 1000, pnlPct: 10.0 },
  { symbol: 'NIFTY24MAY19600CE', quantity: 75, avgPrice: 120, ltp: 135, pnl: 1125, pnlPct: 12.5 },
  { symbol: 'BANKNIFTY24MAY44500CE', quantity: 25, avgPrice: 300, ltp: 280, pnl: -500, pnlPct: -6.67 }
]

const mockTrades = [
  { id: 1, time: '09:15:30', symbol: 'NIFTY24APR19500CE', side: 'BUY', qty: 50, price: 150, status: 'COMPLETE' },
  { id: 2, time: '10:22:15', symbol: 'BANKNIFTY24APR45000PE', side: 'SELL', qty: 25, price: 200, status: 'COMPLETE' },
  { id: 3, time: '11:45:20', symbol: 'NIFTY24MAY19600CE', side: 'BUY', qty: 75, price: 120, status: 'COMPLETE' },
  { id: 4, time: '14:30:45', symbol: 'BANKNIFTY24MAY44500CE', side: 'BUY', qty: 25, price: 300, status: 'PENDING' }
]

const mockAgentStatus = {
  sentiment: { status: 'active', confidence: 0.78, signal: 'bullish' },
  news: { status: 'active', confidence: 0.65, signal: 'neutral' },
  greeks: { status: 'active', confidence: 0.82, signal: 'bearish' },
  rl: { status: 'active', confidence: 0.91, signal: 'bullish' }
}

const mockRiskMetrics = {
  var95: -0.025,
  maxDrawdown: -0.08,
  sharpeRatio: 1.85,
  winRate: 0.72,
  profitFactor: 2.1,
  exposure: 0.64
}

function App() {
  const [isTrading, setIsTrading] = useState(true)
  const [selectedTab, setSelectedTab] = useState('overview')
  const [realTimeData, setRealTimeData] = useState(mockPortfolioData)

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => ({
        ...prev,
        totalValue: prev.totalValue + (Math.random() - 0.5) * 1000,
        dailyPnL: prev.dailyPnL + (Math.random() - 0.5) * 100,
        dailyReturn: prev.dailyReturn + (Math.random() - 0.5) * 0.1
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value)
  }

  const formatPercent = (value) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b bg-card">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Brain className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold">AI Trading Agent</h1>
              </div>
              <Badge variant={isTrading ? "default" : "secondary"} className="animate-pulse">
                {isTrading ? "LIVE TRADING" : "PAUSED"}
              </Badge>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-right">
                <div className="text-sm text-muted-foreground">Portfolio Value</div>
                <div className="text-xl font-bold">{formatCurrency(realTimeData.totalValue)}</div>
              </div>
              
              <div className="flex space-x-2">
                <Button 
                  variant={isTrading ? "destructive" : "default"}
                  size="sm"
                  onClick={() => setIsTrading(!isTrading)}
                >
                  {isTrading ? <Pause className="h-4 w-4 mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                  {isTrading ? "Pause" : "Start"}
                </Button>
                <Button variant="outline" size="sm">
                  <Square className="h-4 w-4 mr-2" />
                  Emergency Stop
                </Button>
                <Button variant="outline" size="sm">
                  <Settings className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="positions">Positions</TabsTrigger>
            <TabsTrigger value="trades">Trades</TabsTrigger>
            <TabsTrigger value="agents">AI Agents</TabsTrigger>
            <TabsTrigger value="risk">Risk</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {formatCurrency(realTimeData.totalPnL)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {formatPercent(realTimeData.totalReturn)} total return
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Daily P&L</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${realTimeData.dailyPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatCurrency(realTimeData.dailyPnL)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {formatPercent(realTimeData.dailyReturn)} today
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{realTimeData.positions}</div>
                  <p className="text-xs text-muted-foreground">
                    {formatCurrency(realTimeData.cash)} cash available
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Risk Score</CardTitle>
                  <Shield className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-yellow-600">Medium</div>
                  <Progress value={mockRiskMetrics.exposure * 100} className="mt-2" />
                  <p className="text-xs text-muted-foreground mt-1">
                    {(mockRiskMetrics.exposure * 100).toFixed(0)}% exposure
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Performance Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Portfolio Performance</CardTitle>
                <CardDescription>Portfolio value vs benchmark over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={mockPerformanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip formatter={(value) => formatCurrency(value)} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#2563eb" 
                      strokeWidth={2}
                      name="Portfolio"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="benchmark" 
                      stroke="#64748b" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="Benchmark"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Trades</CardTitle>
                  <CardDescription>Latest trading activity</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {mockTrades.slice(0, 4).map((trade) => (
                      <div key={trade.id} className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">{trade.symbol}</div>
                          <div className="text-sm text-muted-foreground">
                            {trade.time} • {trade.side} {trade.qty}
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">{formatCurrency(trade.price)}</div>
                          <Badge variant={trade.status === 'COMPLETE' ? 'default' : 'secondary'}>
                            {trade.status}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>AI Agent Status</CardTitle>
                  <CardDescription>Current status of all AI agents</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(mockAgentStatus).map(([agent, data]) => (
                      <div key={agent} className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`w-2 h-2 rounded-full ${data.status === 'active' ? 'bg-green-500' : 'bg-red-500'}`} />
                          <div>
                            <div className="font-medium capitalize">{agent} Agent</div>
                            <div className="text-sm text-muted-foreground">
                              {(data.confidence * 100).toFixed(0)}% confidence
                            </div>
                          </div>
                        </div>
                        <Badge variant={data.signal === 'bullish' ? 'default' : data.signal === 'bearish' ? 'destructive' : 'secondary'}>
                          {data.signal}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Positions Tab */}
          <TabsContent value="positions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Current Positions</CardTitle>
                <CardDescription>All active positions and their P&L</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Symbol</th>
                        <th className="text-right p-2">Quantity</th>
                        <th className="text-right p-2">Avg Price</th>
                        <th className="text-right p-2">LTP</th>
                        <th className="text-right p-2">P&L</th>
                        <th className="text-right p-2">P&L %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mockPositions.map((position, index) => (
                        <tr key={index} className="border-b">
                          <td className="p-2 font-medium">{position.symbol}</td>
                          <td className="p-2 text-right">{position.quantity}</td>
                          <td className="p-2 text-right">{formatCurrency(position.avgPrice)}</td>
                          <td className="p-2 text-right">{formatCurrency(position.ltp)}</td>
                          <td className={`p-2 text-right font-medium ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatCurrency(position.pnl)}
                          </td>
                          <td className={`p-2 text-right ${position.pnlPct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatPercent(position.pnlPct)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Trades Tab */}
          <TabsContent value="trades" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Trade History</CardTitle>
                <CardDescription>Complete trading history for today</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left p-2">Time</th>
                        <th className="text-left p-2">Symbol</th>
                        <th className="text-center p-2">Side</th>
                        <th className="text-right p-2">Quantity</th>
                        <th className="text-right p-2">Price</th>
                        <th className="text-center p-2">Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mockTrades.map((trade) => (
                        <tr key={trade.id} className="border-b">
                          <td className="p-2">{trade.time}</td>
                          <td className="p-2 font-medium">{trade.symbol}</td>
                          <td className="p-2 text-center">
                            <Badge variant={trade.side === 'BUY' ? 'default' : 'destructive'}>
                              {trade.side}
                            </Badge>
                          </td>
                          <td className="p-2 text-right">{trade.qty}</td>
                          <td className="p-2 text-right">{formatCurrency(trade.price)}</td>
                          <td className="p-2 text-center">
                            <Badge variant={trade.status === 'COMPLETE' ? 'default' : 'secondary'}>
                              {trade.status === 'COMPLETE' ? <CheckCircle className="h-3 w-3 mr-1" /> : <Clock className="h-3 w-3 mr-1" />}
                              {trade.status}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* AI Agents Tab */}
          <TabsContent value="agents" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(mockAgentStatus).map(([agent, data]) => (
                <Card key={agent}>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <div className={`w-3 h-3 rounded-full ${data.status === 'active' ? 'bg-green-500' : 'bg-red-500'}`} />
                      <span className="capitalize">{agent} Agent</span>
                    </CardTitle>
                    <CardDescription>
                      {agent === 'sentiment' && 'Analyzes market sentiment from news and social media'}
                      {agent === 'news' && 'Processes financial news and events'}
                      {agent === 'greeks' && 'Calculates options Greeks and risk metrics'}
                      {agent === 'rl' && 'Reinforcement learning trading decisions'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Confidence</span>
                          <span>{(data.confidence * 100).toFixed(0)}%</span>
                        </div>
                        <Progress value={data.confidence * 100} />
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Current Signal</span>
                        <Badge variant={data.signal === 'bullish' ? 'default' : data.signal === 'bearish' ? 'destructive' : 'secondary'}>
                          {data.signal}
                        </Badge>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Status</span>
                        <Badge variant={data.status === 'active' ? 'default' : 'secondary'}>
                          {data.status}
                        </Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Risk Tab */}
          <TabsContent value="risk" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Value at Risk (95%)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-600">
                    {formatPercent(mockRiskMetrics.var95 * 100)}
                  </div>
                  <p className="text-sm text-muted-foreground">Daily VaR</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Max Drawdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-red-600">
                    {formatPercent(mockRiskMetrics.maxDrawdown * 100)}
                  </div>
                  <p className="text-sm text-muted-foreground">Historical maximum</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Sharpe Ratio</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {mockRiskMetrics.sharpeRatio.toFixed(2)}
                  </div>
                  <p className="text-sm text-muted-foreground">Risk-adjusted return</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Win Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">
                    {formatPercent(mockRiskMetrics.winRate * 100)}
                  </div>
                  <p className="text-sm text-muted-foreground">Successful trades</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Profit Factor</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {mockRiskMetrics.profitFactor.toFixed(1)}
                  </div>
                  <p className="text-sm text-muted-foreground">Gross profit / Gross loss</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Portfolio Exposure</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-yellow-600">
                    {formatPercent(mockRiskMetrics.exposure * 100)}
                  </div>
                  <Progress value={mockRiskMetrics.exposure * 100} className="mt-2" />
                  <p className="text-sm text-muted-foreground mt-1">Of total capital</p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Risk Alerts</CardTitle>
                <CardDescription>Current risk warnings and notifications</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Alert>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Medium Risk Level</AlertTitle>
                    <AlertDescription>
                      Portfolio exposure is at 64% of total capital. Consider reducing position sizes.
                    </AlertDescription>
                  </Alert>
                  
                  <Alert>
                    <Eye className="h-4 w-4" />
                    <AlertTitle>Monitoring</AlertTitle>
                    <AlertDescription>
                      High volatility detected in BANKNIFTY options. Increased monitoring active.
                    </AlertDescription>
                  </Alert>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Strategy Performance</CardTitle>
                  <CardDescription>Performance breakdown by strategy</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={[
                      { strategy: 'Sentiment', return: 15.2 },
                      { strategy: 'News', return: 8.7 },
                      { strategy: 'Greeks', return: 12.1 },
                      { strategy: 'RL Agent', return: 18.9 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="strategy" />
                      <YAxis />
                      <Tooltip formatter={(value) => `${value}%`} />
                      <Bar dataKey="return" fill="#2563eb" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Asset Allocation</CardTitle>
                  <CardDescription>Current portfolio allocation</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'NIFTY Options', value: 45, fill: '#2563eb' },
                          { name: 'BANKNIFTY Options', value: 35, fill: '#dc2626' },
                          { name: 'Cash', value: 20, fill: '#16a34a' }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Daily Returns Distribution</CardTitle>
                <CardDescription>Distribution of daily returns over the last 30 days</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={[
                    { return: -3, frequency: 2 },
                    { return: -2, frequency: 5 },
                    { return: -1, frequency: 8 },
                    { return: 0, frequency: 12 },
                    { return: 1, frequency: 15 },
                    { return: 2, frequency: 10 },
                    { return: 3, frequency: 6 },
                    { return: 4, frequency: 3 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="return" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="frequency" stroke="#2563eb" fill="#2563eb" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default App

