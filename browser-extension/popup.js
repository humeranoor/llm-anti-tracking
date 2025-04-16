// Popup script
document.addEventListener('DOMContentLoaded', function() {
  // Update status indicator
  const statusEl = document.getElementById('currentStatus');
  const currentUrlEl = document.getElementById('currentUrl');
  const trackersListEl = document.getElementById('trackersList');
  
  // Set initial loading state
  statusEl.textContent = "CHECKING...";
  statusEl.className = "status checking";
  currentUrlEl.textContent = "Loading...";
  trackersListEl.innerHTML = "<p>Loading tracker information...</p>";
  
  // Get global stats
  chrome.runtime.sendMessage({action: "getStats"}, function(response) {
    if (response && response.stats) {
      document.getElementById('checked').textContent = response.stats.checked;
      document.getElementById('blocked').textContent = response.stats.blocked;
      
      // Update recent list
      const recentList = document.getElementById('recentList');
      recentList.innerHTML = "";
      
      if (response.stats.recentBlocked.length === 0) {
        recentList.innerHTML = "<p>No trackers blocked yet.</p>";
      } else {
        response.stats.recentBlocked.forEach(url => {
          const div = document.createElement('div');
          div.className = 'tracker-item';
          div.textContent = url;
          recentList.appendChild(div);
        });
      }
    } else {
      console.error("Failed to get stats", response);
    }
  });
  
  // Get current tab data
  chrome.runtime.sendMessage({action: "getCurrentTabData"}, function(response) {
    console.log("Got tab data response:", response);
    
    if (response && response.tabUrl) {
      // Update current URL info
      currentUrlEl.textContent = response.tabUrl;
      
      if (response.tabData) {
        // Update current URL status
        if (response.tabData.isTracker) {
          statusEl.textContent = "TRACKING SITE";
          statusEl.className = "status tracker";
        } else {
          statusEl.textContent = "SAFE";
          statusEl.className = "status safe";
        }
        
        // Update trackers for this page
        trackersListEl.innerHTML = "";
        
        if (response.tabData.trackers.length === 0) {
          trackersListEl.innerHTML = "<p>No trackers detected on this page.</p>";
        } else {
          response.tabData.trackers.forEach(tracker => {
            const div = document.createElement('div');
            div.className = 'tracker-item';
            div.innerHTML = `
              <div class="tracker-url">${tracker.url}</div>
              <div class="tracker-confidence">Confidence: ${(tracker.confidence * 100).toFixed(1)}%</div>
            `;
            trackersListEl.appendChild(div);
          });
        }
      } else {
        statusEl.textContent = "UNKNOWN";
        statusEl.className = "status unknown";
        trackersListEl.innerHTML = "<p>No tracker data available.</p>";
      }
    } else {
      // Handle error case
      currentUrlEl.textContent = "Could not determine current URL";
      statusEl.textContent = "ERROR";
      statusEl.className = "status error";
      trackersListEl.innerHTML = "<p>Failed to load tracker information.</p>";
      console.error("Failed to get current tab data", response);
    }
  });
});