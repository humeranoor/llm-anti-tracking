// Background script for LLM Anti-Tracker
const API_ENDPOINT = "http://localhost:5000/predict";
const CACHE_DURATION = 3600000; // Cache results for 1 hour (in ms)

// Simple cache to avoid repeated API calls
const urlCache = new Map();

// Track data per tab
const tabData = new Map();

// Global stats
let stats = {
  checked: 0,
  blocked: 0,
  recentBlocked: []
};

// Load stats from storage
chrome.storage.local.get('stats', (data) => {
  if (data.stats) {
    stats = data.stats;
  }
});

// Check URLs and update data
async function checkUrl(url, tabId) {
  // Check cache first
  const now = Date.now();
  if (urlCache.has(url)) {
    const {result, timestamp} = urlCache.get(url);
    if (now - timestamp < CACHE_DURATION) {
      console.log(`[Cache] ${url.substring(0, 50)}...: ${result.is_tracker ? "BLOCKED" : "ALLOWED"}`);
      updateTabData(tabId, url, result);
      return result;
    }
  }
  
  try {
    // Call the API
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({url: url})
    });
    
    if (!response.ok) throw new Error("API error");
    
    const result = await response.json();
    
    // Cache the result
    urlCache.set(url, {result, timestamp: now});
    
    // Update stats
    stats.checked++;
    if (result.is_tracker) {
      stats.blocked++;
      stats.recentBlocked.unshift(url);
      if (stats.recentBlocked.length > 10) {
        stats.recentBlocked.pop();
      }
      
      // Save updated stats
      chrome.storage.local.set({stats});
    }
    
    // Update tab data
    updateTabData(tabId, url, result);
    
    console.log(`[API] ${url.substring(0, 50)}...: ${result.is_tracker ? "BLOCKED" : "ALLOWED"} (${result.confidence.toFixed(2)})`);
    return result;
  } catch (error) {
    console.error("Error checking URL:", error);
    return {is_tracker: false, confidence: 0};
  }
}

// Update data for a specific tab
function updateTabData(tabId, url, result) {
  if (!tabData.has(tabId)) {
    tabData.set(tabId, {
      url: url,
      isTracker: result.is_tracker,
      confidence: result.confidence,
      trackers: [],
      timestamp: Date.now()
    });
  } else {
    const data = tabData.get(tabId);
    data.url = url;
    data.isTracker = result.is_tracker;
    data.confidence = result.confidence;
    data.timestamp = Date.now();
  }
}

// Track resource requests
chrome.webRequest.onBeforeRequest.addListener(
  async function(details) {
    // Only process if we have a tab ID (main frame or sub-resource)
    if (details.tabId !== -1) {
      const result = await checkUrl(details.url, details.tabId);
      
      // If it's a tracker, add to the tab's tracker list
      if (result.is_tracker) {
        const data = tabData.get(details.tabId) || {
          url: "",
          isTracker: false,
          confidence: 0,
          trackers: [],
          timestamp: Date.now()
        };
        
        // Add to trackers if not already there
        if (!data.trackers.some(t => t.url === details.url)) {
          data.trackers.push({
            url: details.url,
            confidence: result.confidence,
            timestamp: Date.now()
          });
        }
        
        tabData.set(details.tabId, data);
      }
    }
  },
  {urls: ["<all_urls>"]}
);

// Listen for tab updates to check main URLs
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    // Check the main page URL
    await checkUrl(tab.url, tabId);
  }
});

// Clean up tab data when tabs are closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabData.has(tabId)) {
    tabData.delete(tabId);
  }
});

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "getStats") {
    sendResponse({stats: stats});
    return true;
  } 
  
  if (message.action === "getCurrentTabData") {
    // Get data for the current tab
    chrome.tabs.query({active: true, currentWindow: true}, async (tabs) => {
      if (tabs.length > 0) {
        const currentTabId = tabs[0].id;
        const currentTabUrl = tabs[0].url;
        
        // Ensure we have data for this tab
        if (!tabData.has(currentTabId) && currentTabUrl) {
          await checkUrl(currentTabUrl, currentTabId);
        }
        
        const data = tabData.get(currentTabId) || {
          url: currentTabUrl,
          isTracker: false,
          confidence: 0,
          trackers: [],
          timestamp: Date.now()
        };
        
        sendResponse({
          tabData: data, 
          tabUrl: currentTabUrl,
          success: true
        });
      } else {
        sendResponse({
          tabData: null, 
          tabUrl: "",
          success: false,
          error: "No active tab found"
        });
      }
    });
    return true; // Keep the message channel open for the async response
  }
  
  return false;
});