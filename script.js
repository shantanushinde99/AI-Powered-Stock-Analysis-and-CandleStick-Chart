document.addEventListener('DOMContentLoaded', function() {
    // Tab switching
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
      tab.addEventListener('click', function() {
        // Remove active class from all tabs
        tabs.forEach(t => t.classList.remove('active'));
        // Add active class to clicked tab
        this.classList.add('active');
      });
    });
  
    // Time period switching
    const periods = document.querySelectorAll('.period');
    periods.forEach(period => {
      period.addEventListener('click', function() {
        // Remove active class from all periods
        periods.forEach(p => p.classList.remove('active'));
        // Add active class to clicked period
        this.classList.add('active');
      });
    });
  
    // Simulate stock price changes
    function simulatePriceChanges() {
      const priceValue = document.querySelector('.price-value');
      const priceChange = document.querySelector('.price-change');
      
      // Get current price
      let currentPrice = parseFloat(priceValue.textContent.replace('$', ''));
      
      // Random price change between -0.5 and 0.5
      const change = (Math.random() - 0.5) * 0.5;
      const newPrice = (currentPrice + change).toFixed(2);
      
      // Update price display
      priceValue.textContent = `$${newPrice}`;
      
      // Calculate percentage change
      const percentChange = (change / currentPrice * 100).toFixed(2);
      
      // Update change display
      if (change >= 0) {
        priceChange.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="7" y1="17" x2="17" y2="7"/>
            <polyline points="7 7 17 7 17 17"/>
          </svg>
          ${Math.abs(percentChange)}%
        `;
        priceChange.classList.remove('negative');
        priceChange.classList.add('positive');
      } else {
        priceChange.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="7" y1="7" x2="17" y2="17"/>
            <polyline points="17 7 7 7 7 17"/>
          </svg>
          ${Math.abs(percentChange)}%
        `;
        priceChange.classList.remove('positive');
        priceChange.classList.add('negative');
      }
    }
    
    // Simulate price changes every 5 seconds
    setInterval(simulatePriceChanges, 5000);
  });