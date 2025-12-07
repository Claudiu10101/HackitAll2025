import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PenaltiesChart = ({ penaltiesData, formatCurrency }) => {
  const dayKeys = Object.keys(penaltiesData.byDay).sort((a, b) => Number(a) - Number(b));

  // Prepare chart data
  const chartData = dayKeys.map(day => {
    const dayData = penaltiesData.byDay[day];
    return {
      day: `Day ${day}`,
      'First Class Penalty': dayData.FLIGHT_UNFULFILLED_FIRST_CLASS || 0,
      'Business Class Penalty': dayData.FLIGHT_UNFULFILLED_BUSINESS_CLASS || 0,
      'Premium Economy Penalty': dayData.FLIGHT_UNFULFILLED_PREMIUM_ECONOMY_CLASS || 0,
      'Economy Class Penalty': dayData.FLIGHT_UNFULFILLED_ECONOMY_CLASS || 0,
      'Remaining Stock Penalty': dayData.END_OF_GAME_REMAINING_STOCK || 0,
      'Negative Inventory Penalty': dayData.NEGATIVE_INVENTORY || 0,
      'Operational Cost': dayData.OPERATIONAL_COST || 0
    };
  });

  // Custom tooltip formatter
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">{label}</p>
          {payload.map((entry, index) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {formatCurrency(entry.value)}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="image-section">
      <h2>Costs by Day Overview</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart 
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid vertical={false} stroke="none" />
          <XAxis 
            dataKey="day" 
            angle={-45}
            textAnchor="end"
            height={80}
            tick={{ fontSize: 12 }}
            stroke="#666"
          />
          <YAxis 
            tick={{ fontSize: 12 }}
            stroke="#666"
            tickFormatter={(value) => `$${(value / 1000000).toFixed(1)}M`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend 
            verticalAlign="bottom" 
            height={36}
            wrapperStyle={{ paddingTop: '20px' }}
          />
          <Line 
            type="monotone" 
            dataKey="Operational Cost" 
            stroke="#17becf" 
            strokeWidth={3}
            dot={{ r: 4 }}
            activeDot={{ r: 6 }}
          />
          <Line 
            type="monotone" 
            dataKey="Economy Class Penalty" 
            stroke="#1f77b4" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="Business Class Penalty" 
            stroke="#ff7f0e" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="Premium Economy Penalty" 
            stroke="#2ca02c" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="First Class Penalty" 
            stroke="#d62728" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="Remaining Stock Penalty" 
            stroke="#9467bd" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
          <Line 
            type="monotone" 
            dataKey="Negative Inventory Penalty" 
            stroke="#e377c2" 
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PenaltiesChart;
