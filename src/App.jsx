import { useState, useEffect } from 'react'
import axios from 'axios'
import StockChart from './components/StockChart'
import TradingViewWidget from './components/TradingViewWidget'
import { Search, Activity, TrendingUp, DollarSign, BarChart2 } from 'lucide-react'

function App() {
  const [symbol, setSymbol] = useState('AAPL')
  const [searchInput, setSearchInput] = useState('')
  const [data, setData] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingPred, setLoadingPred] = useState(false)
  const [error, setError] = useState(null)
  const [currentStats, setCurrentStats] = useState(null)
  const [viewMode, setViewMode] = useState('analytics') // 'analytics' or 'tradingview'

  const fetchData = async (sym) => {
    setLoading(true)
    setError(null)
    setPredictions(null)
    try {
      const res = await axios.get(`http://127.0.0.1:5000/api/stock/${sym}`)
      setData(res.data)
      setCurrentStats(res.data.stats)
      setSymbol(res.data.symbol)

      // Auto-switch to TradingView for Gold or if user prefers
      if (res.data.symbol === 'XAUUSD' || res.data.symbol === 'GC=F' || res.data.symbol.includes('USD')) {
        setViewMode('tradingview')
      }
    } catch (err) {
      setError(err.response?.data?.error || "Failed to fetch data. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  // Lightweight function to update only current price stats
  const fetchCurrentPrice = async () => {
    if (!symbol) return
    try {
      const res = await axios.get(`http://127.0.0.1:5000/api/price/${symbol}`)
      if (res.data.price) {
        // Update stats with new live price
        setCurrentStats(prev => ({
          ...prev,
          close: res.data.price,
          // approximate high/low update if needed
          high: Math.max(prev?.high || 0, res.data.price),
          low: Math.min(prev?.low || res.data.price, res.data.price)
        }))
      }
    } catch (err) {
      console.error('Failed to update price:', err)
    }
  }

  const fetchPrediction = async (modelType) => {
    if (!data) return;
    setLoadingPred(true)
    try {
      const res = await axios.get(`http://127.0.0.1:5000/api/predict/${symbol}?model=${modelType}`)
      setPredictions(res.data.predictions)
    } catch (err) {
      console.error(err)
      alert("Prediction failed. " + (err.response?.data?.error || ""))
    } finally {
      setLoadingPred(false)
    }
  }

  useEffect(() => {
    fetchData('AAPL')
  }, [])

  // Auto-update removed per user request
  // Price will update when user searches for a symbol or manually refreshes


  useEffect(() => {
    if (!symbol || !data) return;

    // Auto-update price every 3 seconds
    const interval = setInterval(() => {
      fetchCurrentPrice()
    }, 3000)

    return () => clearInterval(interval)
  }, [symbol, data])

  const handleSearch = (e) => {
    e.preventDefault()
    if (searchInput) {
      fetchData(searchInput)
    }
  }

  // Calculate metrics from latest historical data
  const latest = data?.data[data.data.length - 1]
  const previous = data?.data[data.data.length - 2]

  // Use currentStats for live price, fallback to latest historical
  const livePrice = currentStats?.close || latest?.Close || 0
  const liveChange = currentStats && latest ? livePrice - latest.Close : (latest && previous ? latest.Close - previous.Close : 0)
  const liveChangePercent = currentStats && latest ? (liveChange / latest.Close) * 100 : (latest && previous ? ((latest.Close - previous.Close) / previous.Close) * 100 : 0)

  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [priceFlash, setPriceFlash] = useState(false)

  // Update timestamp when currentStats changes
  useEffect(() => {
    if (currentStats) {
      setLastUpdated(new Date())
      // Trigger price flash animation
      setPriceFlash(true)
      setTimeout(() => setPriceFlash(false), 500)
    }
  }, [currentStats])

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '800', background: 'linear-gradient(to right, #3b82f6, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            FutureStocks<span style={{ color: 'white', WebkitTextFillColor: 'white' }}>Prediction</span>
          </h1>
          <p style={{ margin: '0.2rem 0 0 0', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            AI-Powered Market Analytics
            {data && (
              <span style={{ marginLeft: '1rem' }}>
                • Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <form onSubmit={handleSearch} style={{ position: 'relative', width: '300px' }}>
            <input
              type="text"
              className="input-field"
              placeholder="Search Symbol (e.g. TSLA, BTC, ETH)"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              style={{ paddingLeft: '2.5rem' }}
            />
            <Search size={18} style={{ position: 'absolute', left: '0.8rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
          </form>
        </div>
      </header>

      {error && (
        <div style={{ color: 'var(--danger)', marginBottom: '1rem', padding: '1rem', border: '1px solid var(--danger)', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.1)' }}>
          {error}
        </div>
      )}

      {data?.warning && (
        <div style={{ color: '#f59e0b', marginBottom: '1rem', padding: '1rem', border: '1px solid #f59e0b', borderRadius: '8px', background: 'rgba(245, 158, 11, 0.1)' }}>
          ⚠️ {data.warning}
        </div>
      )}

      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', margin: '4rem' }}>
          <div className="spinner"></div>
        </div>
      ) : data ? (
        <>
          <div className="grid-dashboard" style={{ marginTop: '0', marginBottom: '2rem' }}>
            <div className={`card flex-center ${priceFlash ? 'price-update' : ''}`} style={{ justifyContent: 'space-between', position: 'relative', overflow: 'hidden' }}>
              {/* Live indicator pulse */}
              <div style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}>
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--success)',
                  animation: 'pulse 2s infinite'
                }}></div>
                <span style={{ fontSize: '0.75rem', color: 'var(--success)', fontWeight: '600' }}>LIVE</span>
              </div>

              <div>
                <div className="metric-label">Current Price</div>
                <div className="metric-value" style={{
                  fontSize: '2.5rem',
                  background: 'linear-gradient(135deg, #3b82f6, #818cf8)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  transition: 'all 0.3s ease'
                }}>
                  ${livePrice.toFixed(2)}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                  {data.symbol}
                </div>
              </div>
              <div className="flex-center" style={{
                color: liveChange >= 0 ? 'var(--success)' : 'var(--danger)',
                background: liveChange >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                padding: '1rem',
                borderRadius: '12px',
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  {liveChange >= 0 ? <TrendingUp size={28} /> : <TrendingUp size={28} style={{ transform: 'scaleY(-1)' }} />}
                  <span style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>{liveChangePercent.toFixed(2)}%</span>
                </div>
                <span style={{ fontSize: '0.85rem', opacity: 0.8 }}>
                  ${Math.abs(liveChange).toFixed(2)}
                </span>
              </div>
            </div >

            {/* Trading Signal Card */}
            {
              data.signals && (
                <div className="card" style={{ gridColumn: '1 / -1' }}> {/* Span full width */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ margin: 0 }}>Technical Analysis</h3>
                    <div style={{
                      padding: '0.5rem 1rem',
                      borderRadius: '8px',
                      background: data.signals.signal.includes('BUY') ? 'var(--success)' : data.signals.signal.includes('SELL') ? 'var(--danger)' : 'var(--text-secondary)',
                      color: 'white',
                      fontWeight: 'bold',
                      boxShadow: '0 0 15px rgba(0,0,0,0.2)'
                    }}>
                      {data.signals.signal} ({data.signals.confidence})
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '2rem' }}>

                    {/* Detailed Indicators Table */}
                    <div style={{ background: 'rgba(255,255,255,0.02)', borderRadius: '12px', padding: '1rem' }}>
                      <h4 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>Indicator Breakdown</h4>
                      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
                        <thead>
                          <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                            <th style={{ padding: '0.5rem' }}>Indicator</th>
                            <th style={{ padding: '0.5rem' }}>Value</th>
                            <th style={{ padding: '0.5rem' }}>Signal</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.signals.analysis && data.signals.analysis.map((item, index) => (
                            <tr key={index} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                              <td style={{ padding: '0.8rem 0.5rem' }}>
                                <div style={{ fontWeight: '500' }}>{item.name}</div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{item.condition}</div>
                              </td>
                              <td style={{ padding: '0.8rem 0.5rem', fontFamily: 'monospace' }}>{item.value}</td>
                              <td style={{ padding: '0.8rem 0.5rem', fontWeight: 'bold', color: item.signal.includes('BUY') ? 'var(--success)' : item.signal.includes('SELL') ? 'var(--danger)' : 'var(--text-secondary)' }}>
                                {item.signal}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Trade Setup Panel */}
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                      <h4 style={{ marginTop: 0, color: 'var(--text-secondary)' }}>
                        {data.signals.strategy ? data.signals.strategy : "Trade Setup"} <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>(Experimental)</span>
                      </h4>

                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                        <div style={{ background: 'rgba(59, 130, 246, 0.1)', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                          <div style={{ color: 'var(--accent)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Entry Price</div>
                          <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>${data.signals.entry_price.toFixed(2)}</div>
                        </div>

                        <div style={{ background: 'rgba(16, 185, 129, 0.1)', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                          <div style={{ color: 'var(--success)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Take Profit</div>
                          <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>${data.signals.take_profit.toFixed(2)}</div>
                        </div>

                        <div style={{ background: 'rgba(239, 68, 68, 0.1)', padding: '1rem', borderRadius: '12px', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                          <div style={{ color: 'var(--danger)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Stop Loss</div>
                          <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>${data.signals.stop_loss.toFixed(2)}</div>
                        </div>

                        <div style={{ background: 'rgba(255, 255, 255, 0.05)', padding: '1rem', borderRadius: '12px', border: '1px solid var(--border)' }}>
                          <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '0.25rem' }}>Consensus Score</div>
                          <div style={{ fontSize: '1.25rem', fontWeight: 'bold' }}>{data.signals.score > 0 ? '+' : ''}{data.signals.score}/8</div>
                        </div>
                      </div>

                      <div style={{ fontSize: '0.8rem', opacity: 0.6, marginTop: 'auto', fontStyle: 'italic' }}>
                        *Signals are based on historical data analysis (RSI, MACD, SMA, Bollinger Bands) and should not be taken as financial advice.
                      </div>
                    </div>
                  </div>
                </div>
              )
            }

            <div className="card">
              <div className="metric-label">RSI (14)</div>
              <div className="metric-value" style={{ color: latest.RSI > 70 ? 'var(--danger)' : latest.RSI < 30 ? 'var(--success)' : 'var(--text-primary)' }}>
                {latest.RSI ? latest.RSI.toFixed(2) : 'N/A'}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {latest.RSI > 70 ? 'Overbought' : latest.RSI < 30 ? 'Oversold' : 'Neutral'}
              </div>
            </div>

            <div className="card">
              <div className="metric-label">Volume</div>
              <div className="metric-value">{(latest.Volume / 1000000).toFixed(2)}M</div>
            </div>

            <div className="card">
              <div className="metric-label">24h High</div>
              <div className="metric-value" style={{ color: 'var(--success)' }}>
                ${data.stats ? data.stats.high.toFixed(2) : latest.High.toFixed(2)}
              </div>
            </div>

            <div className="card">
              <div className="metric-label">24h Low</div>
              <div className="metric-value" style={{ color: 'var(--danger)' }}>
                ${data.stats ? data.stats.low.toFixed(2) : latest.Low.toFixed(2)}
              </div>
            </div>

            <div className="card">
              <div className="metric-label">Open</div>
              <div className="metric-value">
                ${data.stats ? data.stats.open.toFixed(2) : latest.Open.toFixed(2)}
              </div>
            </div>
          </div >



          <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
            <button
              className="btn"
              style={{
                background: viewMode === 'analytics' ? 'var(--accent)' : 'var(--bg-card)',
                border: viewMode === 'analytics' ? 'none' : '1px solid var(--border)'
              }}
              onClick={() => setViewMode('analytics')}
            >
              <Activity size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              AI Analytics
            </button>
            <button
              className="btn"
              style={{
                background: viewMode === 'tradingview' ? 'var(--accent)' : 'var(--bg-card)',
                border: viewMode === 'tradingview' ? 'none' : '1px solid var(--border)'
              }}
              onClick={() => setViewMode('tradingview')}
            >
              <BarChart2 size={18} style={{ marginRight: '0.5rem', verticalAlign: 'text-bottom' }} />
              Live Chart
            </button>
          </div>

          {
            viewMode === 'analytics' ? (
              <StockChart data={data.data} predictions={predictions} symbol={data.symbol} company={data.company} />
            ) : (
              <TradingViewWidget symbol={data.symbol} />
            )
          }

          <div className="card" style={{ marginTop: '2rem' }}>
            <div className="header" style={{ marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>AI Prediction</h3>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <button className="btn" disabled={loadingPred} onClick={() => fetchPrediction('linear')}>
                  {loadingPred && predictions?.length === 0 ? 'Thinking...' : 'Linear Regression'}
                </button>
                <button className="btn" disabled={loadingPred} style={{ background: '#8b5cf6' }} onClick={() => fetchPrediction('lstm')}>
                  {loadingPred ? 'Training & Predicting...' : 'LSTM (Deep Learning)'}
                </button>
              </div>
            </div>
            <p style={{ color: 'var(--text-secondary)' }}>
              Generate a 7-day forecast using our advanced machine learning models.
              Linear Regression provides a simple trend following, while LSTM (Long Short-Term Memory) captures complex sequences.
              Note: LSTM prediction may take a few seconds to train on the fly.
            </p>
          </div>
        </>
      ) : null
      }
    </div >
  )
}

export default App
