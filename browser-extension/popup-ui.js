// popup-ui.js - Contains UI interaction code for the popup

// Tab switching logic
document.addEventListener('DOMContentLoaded', function() {
  // Set up tab switching
  document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
      // Remove active class from all buttons and content
      document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      
      // Add active class to clicked button and associated content
      button.classList.add('active');
      document.getElementById(button.dataset.tab + 'Tab').classList.add('active');
    });
  });
  
  // Debug connection button
  document.getElementById('debugBtn').addEventListener('click', () => {
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url: 'https://example.com'})
    })
    .then(response => {
      if (response.ok) return response.json();
      throw new Error('API server not responding correctly');
    })
    .then(data => {
      alert('API server is running and responding correctly!');
    })
    .catch(error => {
      alert('Error connecting to API server: ' + error.message);
    });
  });
});