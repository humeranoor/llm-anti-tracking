# realistic_data_generator.py
import pandas as pd
import random
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

# Real tracker patterns from EasyPrivacy
TRACKER_PATTERNS = [
    r'/ads?/', r'track(ing|er)', r'analytics', r'pixel', r'tagmanager',
    r'doubleclick.net', r'googleadservices.com', r'googletagmanager.com',
    r'fbcdn.net', r'facebook.com/tr/', r'linkedin.com/analytics',
    r'/gtm.js', r'hotjar.com', r'pardot.com', r'mouseflow.com'
]

# Real popular domains from Majestic Million (top 10k)
TOP_DOMAINS = [
    'google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
    'reddit.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'microsoft.com',
    'apple.com', 'netflix.com', 'cloudflare.com', 'adobe.com', 'cdn.shopify.com'
]

# Common static file extensions
STATIC_EXTS = ['.css', '.js', '.png', '.jpg', '.svg', '.woff2', '.ico']

def generate_realistic_url(is_tracker):
    """Generate URLs that mimic real-world tracking/static resources"""
    domain = random.choice(TOP_DOMAINS)
    path_depth = random.randint(1, 4)
    
    # Generate path segments
    path = []
    for _ in range(path_depth):
        segment = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3,8)))
        path.append(segment)
    
    # Add file extension for static resources
    if random.random() > 0.7 and not is_tracker:
        path[-1] += random.choice(STATIC_EXTS)
    
    # Add tracking parameters
    query = []
    if is_tracker:
        # Real tracking parameters
        params = ['utm_', 'cid', 'sessionid', 'ref=', 'affiliate', 'campaign']
        for _ in range(random.randint(1,3)):
            key = random.choice(params) + ''.join(random.choices('0123456789', k=3))
            val = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
            query.append(f"{key}={val}")
    
    # Construct URL
    url = f"https://{domain}/{'/'.join(path)}"
    if query:
        url += "?" + "&".join(query)
    
    # Add tracker patterns
    if is_tracker:
        pattern = random.choice(TRACKER_PATTERNS)
        if pattern.startswith('/'):
            url += pattern
        else:
            url = url.replace(domain, pattern)
    
    return url

def generate_dataset(n=15000):
    data = []
    start_time = datetime.now() - timedelta(days=30)
    
    for _ in range(n):
        is_tracker = random.random() < 0.35  # 35% trackers
        timestamp = start_time + timedelta(seconds=random.randint(0, 2592000))
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "url": generate_realistic_url(is_tracker),
            "status_code": random.choices(
                [200, 404, 302, 500], 
                weights=[0.85, 0.05, 0.08, 0.02]
            )[0],
            "content_type": random.choice([
                "text/html", "application/javascript", 
                "image/png", "text/css"
            ]),
            "is_tracker": int(is_tracker)
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("train_data.csv", index=False)
    print(f"Generated {len(df)} requests")
    print("Sample trackers:")
    print(df[df['is_tracker'] == 1].sample(3)['url'].tolist())