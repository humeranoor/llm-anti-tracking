from proxy.http.server import HttpProxyBasePlugin, HttpWebServerPlugin, HttpWebServerRequestHandler
from proxy.http.parser import HttpParser
import requests
import json
import time

class TrackingDetectorPlugin(HttpProxyBasePlugin):
    """Plugin to detect and block tracking URLs."""
    
    API_ENDPOINT = "http://localhost:5000/predict"
    cache = {}  # Simple in-memory cache
    CACHE_EXPIRY = 3600  # 1 hour in seconds
    
    def before_upstream_connection(self, request: HttpParser) -> None:
        url = request.url.decode()
        
        # Check cache
        now = time.time()
        if url in self.cache and (now - self.cache[url]['timestamp']) < self.CACHE_EXPIRY:
            result = self.cache[url]['result']
            print(f"[Cache] {url[:50]}... : {result['is_tracker']}")
            if result['is_tracker']:
                # Block the request by closing the client connection
                self.client_conn.close()
                return
        
        # Call the API to check if it's a tracker
        try:
            response = requests.post(
                self.API_ENDPOINT,
                json={"url": url},
                timeout=1  # Short timeout to avoid hanging
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Cache the result
                self.cache[url] = {
                    'result': result,
                    'timestamp': now
                }
                
                print(f"[API] {url[:50]}... : {result['is_tracker']} ({result['confidence']:.2f})")
                
                # If it's a tracker, block the request
                if result['is_tracker']:
                    self.client_conn.close()
        except Exception as e:
            print(f"Error checking URL: {e}")

if __name__ == "__main__":
    proxy_server = HttpWebServerPlugin(
        HandlerClass=HttpWebServerRequestHandler,
        ServerClass=HttpWebServerPlugin,
        port=8899,
        plugins=[TrackingDetectorPlugin()]
    )
    proxy_server.start()
    print("Proxy server running on port 8899")