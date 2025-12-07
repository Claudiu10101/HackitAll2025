import React, { useState, useEffect, useRef } from 'react';
import PenaltiesChart from './PenaltiesChart.jsx';
import PenaltiesBarChart from './PenaltiesBarChart.jsx';
import './Dashboard.css';

function Dashboard() {
  const [penaltiesData, setPenaltiesData] = useState(null);
  const [currentDay, setCurrentDay] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const prevDayCountRef = useRef(0);
  const currentDayRef = useRef(0);

  // Keep a ref of the selected day to avoid stale closures in the poller
  useEffect(() => {
    currentDayRef.current = currentDay;
  }, [currentDay]);

  useEffect(() => {
    let isMounted = true;

    const parseCsv = (csvText) => {
      const lines = csvText.trim().split('\n');
      const result = {};
      for (let i = 1; i < lines.length; i++) { // Skip header
        const [day, cost] = lines[i].split(',');
        if (day && cost) {
          result[day] = parseFloat(cost);
        }
      }
      return result;
    };

    const loadPenalties = async (isInitial = false) => {
      try {
        const timestamp = Date.now();
        
        // Fetch penalties data
        const penaltiesUrl = `/penalties/penalties_summary.json?t=${timestamp}`;
        const penaltiesResponse = await fetch(penaltiesUrl, { cache: 'no-store' });
        if (!penaltiesResponse.ok) throw new Error('Failed to fetch penalties data');
        const data = await penaltiesResponse.json();
        
        // Fetch operational costs data
        const costsUrl = `/costs/costs_by_day.csv?t=${timestamp}`;
        try {
          const costsResponse = await fetch(costsUrl, { cache: 'no-store' });
          if (costsResponse.ok) {
            const csvText = await costsResponse.text();
            console.log('CSV loaded, first 200 chars:', csvText.substring(0, 200));
            const operationalCosts = parseCsv(csvText);
            console.log('Parsed operational costs:', operationalCosts);
            console.log('Sample days from penalties:', Object.keys(data.byDay).slice(0, 5));
            
            // Merge operational costs into penalties data
            // Subtract penalty costs from total cost to get true operational cost
            for (const [day, totalCost] of Object.entries(operationalCosts)) {
              if (!data.byDay[day]) {
                data.byDay[day] = {};
              }
              
              // Calculate penalty sum for this day
              const dayPenalties = data.byDay[day];
              const penaltySum = Object.entries(dayPenalties)
                .filter(([key]) => key !== 'OPERATIONAL_COST')
                .reduce((sum, [, value]) => sum + value, 0);
              
              // True operational cost = total cost - penalties
              const trueOperationalCost = totalCost - penaltySum;
              data.byDay[day].OPERATIONAL_COST = trueOperationalCost;
            }
            
            // Calculate total operational cost
            const totalOperationalCost = Object.entries(operationalCosts).reduce((sum, [day, totalCost]) => {
              const dayPenalties = data.byDay[day];
              if (dayPenalties) {
                const penaltySum = Object.entries(dayPenalties)
                  .filter(([key]) => key !== 'OPERATIONAL_COST')
                  .reduce((s, [, value]) => s + value, 0);
                return sum + (totalCost - penaltySum);
              }
              return sum + totalCost;
            }, 0);
            
            data.operationalCost = totalOperationalCost;
            console.log('Total operational cost:', data.operationalCost);
            console.log('Sample merged day 0:', data.byDay['0']);
          } else {
            console.warn('Failed to fetch costs CSV:', costsResponse.status);
          }
        } catch (costsErr) {
          console.error('Error fetching/parsing costs:', costsErr);
        }
        
        if (!isMounted) return;

        setPenaltiesData(data);
        setLoading(false);
        setError(null);

        // Detect new days and keep selection in range
        const dayKeys = Object.keys(data.byDay)
          .filter(day => day !== '30')
          .sort((a, b) => Number(a) - Number(b));
        const dayCount = dayKeys.length;
        const prevDayCount = prevDayCountRef.current;

        if (dayCount > prevDayCount) {
          setCurrentDay(dayCount - 1); // jump to latest day when a new one appears
        } else if (currentDayRef.current >= dayCount) {
          setCurrentDay(Math.max(dayCount - 1, 0));
        }

        prevDayCountRef.current = dayCount;
      } catch (err) {
        if (!isMounted) return;
        setError(err.message);
        if (isInitial) setLoading(false);
      }
    };

    loadPenalties(true);
    const intervalId = setInterval(() => loadPenalties(false), 10000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount);
  };

  const calculatePredictedCost = () => {
    const dayKeys = Object.keys(penaltiesData.byDay)
      .filter(day => day !== '30') // Exclude day 30 (END_OF_GAME_REMAINING_STOCK)
      .sort((a, b) => Number(a) - Number(b));

    // Use only fully completed days (exclude the most recent ongoing day)
    const completedDays = dayKeys.slice(0, Math.max(dayKeys.length - 1, 0));

    const totalCostUpToNow = completedDays.reduce((sum, day) => {
      const dayTotal = Object.values(penaltiesData.byDay[day] || {}).reduce((s, v) => s + v, 0);
      return sum + dayTotal;
    }, 0);

    const daysCompleted = completedDays.length || 1; // avoid divide by zero
    const averageCostPerDay = totalCostUpToNow / daysCompleted;
    const totalDays = 30;
    const remainingDays = totalDays - completedDays.length;
    const predictedFinalCost = totalCostUpToNow + (averageCostPerDay * remainingDays);

    return {
      average: averageCostPerDay,
      predicted: predictedFinalCost,
      daysCompleted: completedDays.length,
      remainingDays
    };
  };

  const dayKeys = penaltiesData
    ? Object.keys(penaltiesData.byDay)
        .filter(day => day !== '30')
        .sort((a, b) => Number(a) - Number(b))
    : [];

  // Ensure currentDay stays within bounds after data refreshes
  useEffect(() => {
    if (!penaltiesData) return;
    if (currentDay >= dayKeys.length) {
      setCurrentDay(Math.max(dayKeys.length - 1, 0));
    }
  }, [penaltiesData, dayKeys.length, currentDay]);

  if (loading) {
    return <div className="dashboard-loading">Loading dashboard...</div>;
  }

  if (error) {
    return <div className="dashboard-error">Error: {error}</div>;
  }

  const isGameComplete = false;
  const prediction = calculatePredictedCost();

  return (
    <div className="dashboard">
      {/* Total Cost Section */}
      <div className="cost-sections">
        <div className="total-cost-section">
          <h2>Total Cost</h2>
          <div className="total-cost-amount">
            {formatCurrency(penaltiesData.adjustedTotalCost)}
          </div>
          <p className="cost-subtitle">Days: {prediction.daysCompleted}/30</p>
        </div>

        {!isGameComplete && (
          <div className="predicted-cost-section">
            <h2>Predicted Final Cost</h2>
            <div className="predicted-cost-amount">
              {formatCurrency(prediction.predicted)}
            </div>
            <p className="cost-subtitle">
              Avg: {formatCurrency(prediction.average)}/day
            </p>
          </div>
        )}
      </div>

      {/* Content Container for side-by-side layout */}
      <div className="content-container">
        <PenaltiesChart 
          penaltiesData={penaltiesData}
          formatCurrency={formatCurrency}
        />
        
        <PenaltiesBarChart 
          penaltiesData={penaltiesData}
          formatCurrency={formatCurrency}
        />
      </div>
    </div>
  );
}

export default Dashboard;