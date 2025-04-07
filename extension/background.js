chrome.webRequest.onBeforeRequest.addListener(
  async (details) => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: details.url })
      });
      const { is_tracker } = await response.json();
      return { cancel: is_tracker };
    } catch (error) {
      return { cancel: false }; // Fail open
    }
  },
  { urls: ["<all_urls>"] },
  ["blocking"]
);