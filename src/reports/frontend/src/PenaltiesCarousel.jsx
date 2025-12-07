import React from 'react';

const PenaltiesCarousel = ({ penaltiesData, currentDay, setCurrentDay, formatCurrency }) => {
  const dayKeys = Object.keys(penaltiesData.byDay).sort((a, b) => Number(a) - Number(b));
  const currentDayKey = dayKeys[currentDay];
  const currentDayData = penaltiesData.byDay[currentDayKey];

  const nextDay = () => {
    if (currentDay < dayKeys.length - 1) {
      setCurrentDay(currentDay + 1);
    }
  };

  const prevDay = () => {
    if (currentDay > 0) {
      setCurrentDay(currentDay - 1);
    }
  };

  return (
    <div className="carousel-section">
      <h2>Daily Costs Breakdown</h2>
      <div className="carousel">
        <button 
          className="carousel-btn prev" 
          onClick={prevDay}
          disabled={currentDay === 0}
        >
          ‹
        </button>
        
        <div className="carousel-content">
          <h3>Day {currentDayKey}</h3>
          <div className="penalties-list">
            {currentDayData && Object.entries(currentDayData)
              .sort(([, amountA], [, amountB]) => amountB - amountA)
              .map(([type, amount]) => (
                <div key={type} className="penalty-item">
                  <span className="penalty-type">{type.replace(/_/g, ' ')}</span>
                  <span className="penalty-amount">{formatCurrency(amount)}</span>
                </div>
              ))}
          </div>
          <div className="day-total">
            <strong>Day Total: </strong>
            {formatCurrency(
              Object.values(currentDayData || {}).reduce((sum, val) => sum + val, 0)
            )}
          </div>
        </div>
        
        <button 
          className="carousel-btn next" 
          onClick={nextDay}
          disabled={currentDay === dayKeys.length - 1}
        >
          ›
        </button>
      </div>
      
      <div className="carousel-indicators">
        {dayKeys.map((day, index) => (
          <span 
            key={day}
            className={`indicator ${index === currentDay ? 'active' : ''}`}
            onClick={() => setCurrentDay(index)}
          />
        ))}
      </div>
    </div>
  );
};

export default PenaltiesCarousel;
