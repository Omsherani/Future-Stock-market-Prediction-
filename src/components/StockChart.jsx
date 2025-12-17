import React from 'react';
import Plot from 'react-plotly.js';

const StockChart = ({ data, predictions, symbol, company }) => {
    if (!data || data.length === 0) return null;

    const dates = data.map(d => d.Date);
    const close = data.map(d => d.Close);
    const sma20 = data.map(d => d.SMA_20);
    const sma50 = data.map(d => d.SMA_50);

    const traces = [
        {
            x: dates,
            y: close,
            type: 'scatter',
            mode: 'lines',
            name: 'Close Price',
            line: { color: '#3b82f6', width: 2 }
        },
        {
            x: dates,
            y: sma20,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 20',
            line: { color: '#10b981', width: 1.5 },
            opacity: 0.7
        },
        {
            x: dates,
            y: sma50,
            type: 'scatter',
            mode: 'lines',
            name: 'SMA 50',
            line: { color: '#f59e0b', width: 1.5 },
            opacity: 0.7
        }
    ];

    if (predictions && predictions.length > 0) {
        const predDates = predictions.map(p => p.date);
        const predPrices = predictions.map(p => p.price);

        // Connect last data point to first prediction
        // This is purely visual
        const lastDataDate = dates[dates.length - 1];
        const lastDataPrice = close[close.length - 1];

        const connectX = [lastDataDate, predDates[0]];
        const connectY = [lastDataPrice, predPrices[0]];

        traces.push({
            x: connectX,
            y: connectY,
            mode: 'lines',
            line: { color: '#ef4444', dash: 'dot', width: 2 },
            showlegend: false,
            hoverinfo: 'skip'
        });

        traces.push({
            x: predDates,
            y: predPrices,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Prediction',
            line: { color: '#ef4444', dash: 'dot', width: 2 },
            marker: { size: 6 }
        });
    }

    const chartTitle = company && company !== symbol ? `${symbol} - ${company}` : symbol;

    return (
        <div className="card" style={{ height: '500px', width: '100%' }}>
            <Plot
                data={traces}
                layout={{
                    autosize: true,
                    title: {
                        text: chartTitle,
                        font: { color: '#f3f4f6', size: 18, family: 'Inter, sans-serif' },
                        x: 0.5,
                        xanchor: 'center'
                    },
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { color: '#f3f4f6', family: 'Inter, sans-serif' },
                    xaxis: {
                        gridcolor: '#374151',
                        showgrid: true,
                        zerolinecolor: '#374151'
                    },
                    yaxis: {
                        gridcolor: '#374151',
                        showgrid: true,
                        zerolinecolor: '#374151'
                    },
                    margin: { t: 60, r: 20, l: 50, b: 40 },
                    legend: { orientation: 'h', y: 1.15, x: 0.5, xanchor: 'center' },
                    hovermode: 'x unified'
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true}
            />
        </div>
    );
};

export default StockChart;
