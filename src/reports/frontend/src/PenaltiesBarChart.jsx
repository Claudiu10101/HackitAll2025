import React from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const PenaltiesBarChart = ({ penaltiesData, formatCurrency }) => {
  // Calculate average spending for each penalty type using only completed days (exclude latest in-progress day)
  const dayKeys = Object.keys(penaltiesData.byDay)
    .filter(day => day !== '30')
    .sort((a, b) => Number(a) - Number(b));
  const completedDays = dayKeys.slice(0, Math.max(dayKeys.length - 1, 0));
  
  const penaltyTypes = {
    'OPERATIONAL_COST': 'Operational Cost',
    'FLIGHT_UNFULFILLED_FIRST_CLASS': 'First Class Penalty',
    'FLIGHT_UNFULFILLED_BUSINESS_CLASS': 'Business Class Penalty',
    'FLIGHT_UNFULFILLED_PREMIUM_ECONOMY_CLASS': 'Premium Economy Penalty',
    'FLIGHT_UNFULFILLED_ECONOMY_CLASS': 'Economy Class Penalty',
    'END_OF_GAME_REMAINING_STOCK': 'Remaining Stock Penalty',
    'NEGATIVE_INVENTORY': 'Negative Inventory Penalty'
  };

  const averagesByType = {};
  
  Object.entries(penaltyTypes).forEach(([key, label]) => {
    let total = 0;
    let count = 0;

    completedDays.forEach(day => {
      if (penaltiesData.byDay[day][key]) {
        total += penaltiesData.byDay[day][key];
        count++;
      }
    });

    averagesByType[label] = count > 0 ? total / count : 0;
  });

  const chartData = Object.entries(averagesByType).map(([type, avg]) => ({
    name: type,
    value: avg
  }));

  const COLORS = ['#17becf', '#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#e377c2'];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">{payload[0].payload.name}</p>
          <p style={{ color: payload[0].color }}>
            Avg: {formatCurrency(payload[0].value)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="carousel-section">
      <h2>Average Spending by Cost Type</h2>
      <ResponsiveContainer width="100%" height={400}>
        <PieChart>
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, value }) => `${name}: ${formatCurrency(value)}`}
            outerRadius={120}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PenaltiesBarChart;
