import React, { useEffect, useRef, memo } from 'react';

function TradingViewWidget({ symbol }) {
    const container = useRef();

    useEffect(() => {
        // Clear previous widget
        if (container.current) {
            container.current.innerHTML = '';
        }

        const script = document.createElement("script");
        script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
        script.type = "text/javascript";
        script.async = true;

        // Map common symbols to TradingView format
        let tvSymbol = symbol;
        if (symbol === 'XAUUSD' || symbol === 'GC=F' || symbol === 'GOLD') {
            tvSymbol = 'OANDA:XAUUSD';
        } else if (['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'DOGE'].includes(symbol)) {
            tvSymbol = `BINANCE:${symbol}USDT`;
        } else if (!symbol.includes(':')) {
            // Heuristic: If it looks like a crypto pair (contains USDT/USD) or is known crypto
            tvSymbol = `NASDAQ:${symbol}`;
        }

        script.innerHTML = JSON.stringify({
            "autosize": true,
            "symbol": tvSymbol,
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "enable_publishing": false,
            "allow_symbol_change": true,
            "calendar": false,
            "support_host": "https://www.tradingview.com"
        });

        container.current.appendChild(script);
    }, [symbol]);

    return (
        <div className="card" style={{ height: '600px', padding: 0, overflow: 'hidden' }}>
            <div className="tradingview-widget-container" ref={container} style={{ height: "100%", width: "100%" }}>
                <div className="tradingview-widget-container__widget" style={{ height: "100%", width: "100%" }}></div>
            </div>
        </div>
    );
}

export default memo(TradingViewWidget);
