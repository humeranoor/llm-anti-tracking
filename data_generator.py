# data_generator.py

'''
Use this script in case you want to create data

By default:
1. It will create 15000 urls of whhich 35% are trackers
2. it will create a file: training_data.csv with urls and labels.

To change any of above make following changes:

1. when calling: generate_dataset specify n as number of datapoints and tracker_ratio as percentage of trackers
2. change filename under main.


For better evaluation, ensure that training and testing files are created separately.
'''
import pandas as pd
import random
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import random

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

def load_easyprivacy_rules():
    """Load EasyPrivacy rules from files or use fallback rules"""
    try:
        # General tracking patterns
        with open("easylist/easyprivacy/easyprivacy_general.txt") as f:
            general_rules = [line.strip() for line in f if line.strip() and not line.startswith('!')]
        
        # Specific tracker domains
        with open("easylist/easyprivacy/easyprivacy_specific.txt") as f:
            specific_rules = [line.strip() for line in f if line.strip() and not line.startswith('!')]
        
        easyprivacy_rules = general_rules + specific_rules
        print(f"Loaded {len(easyprivacy_rules)} EasyPrivacy rules")

    except FileNotFoundError:
        print("EasyPrivacy files not found. Using fallback rules.")
        easyprivacy_rules = [
            r"google-analytics\.com",
            r"doubleclick\.net",
            r"facebook\.net",
            r"googletagmanager\.com",
            r"/analytics\.js",
            r"/tr/",
            r"/tracking-pixel",
            r"adform\.net",
            r"adsrvr\.org",
            r"adroll\.com",
            r"amazon-adsystem\.com",
            r"amplitude\.com",
            r"bkrtx\.com",
            r"chartbeat\.com",
            r"clarity\.ms",
            r"clicktale\.net",
            r"crazyegg\.com",
            r"criteo\.com",
            r"glancecdn\.net",
            r"googleadservices\.com",
            r"googlesyndication\.com",
            r"heapanalytics\.com",
            r"media\.net",
            r"mixpanel\.com",
            r"optimizely\.com",
            r"quantserve\.com",
            r"rubiconproject\.com",
            r"segment\.com",
            r"taboola\.com",
            r"tiktok\.com/i18n/pixel"
        ]
    
    return easyprivacy_rules

def generate_realistic_url(is_tracker, easyprivacy_rules):
    """Generate URLs that mimic real-world tracking/static resources"""
    # For tracker URLs, use EasyPrivacy rules directly
    if is_tracker:
        # Select a random rule from EasyPrivacy that is a domain pattern
        domain_rules = [rule for rule in easyprivacy_rules if '/' not in rule]
        path_rules = [rule for rule in easyprivacy_rules if '/' in rule]
        
        # Choose between domain or path-based rule
        if domain_rules and random.random() < 0.6:  # 60% chance of domain-based tracker
            tracker_domain = random.choice(domain_rules)
            # Clean up the domain pattern to make it a valid domain
            tracker_domain = tracker_domain.replace('||', '').replace('^', '')
            
            # Generate path
            path_depth = random.randint(1, 3)
            path = []
            for _ in range(path_depth):
                segment = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3,8)))
                path.append(segment)
            
            url = f"https://{tracker_domain}/{'/'.join(path)}"
            
            # Add tracking parameters
            params = ['utm_', 'cid', 'sessionid', 'ref=', 'affiliate', 'campaign']
            query = []
            for _ in range(random.randint(1,3)):
                key = random.choice(params) + ''.join(random.choices('0123456789', k=3))
                val = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
                query.append(f"{key}={val}")
            
            if query:
                url += "?" + "&".join(query)
                
        else:  # Path-based tracker
            domain = random.choice(TOP_DOMAINS)
            
            # If we have path rules, use one
            if path_rules:
                path_rule = random.choice(path_rules)
                # Remove filters and clean up
                path_rule = path_rule.replace('||', '').replace('^', '')
                
                # Generate additional path segments
                extra_path = []
                for _ in range(random.randint(0, 2)):
                    segment = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3,8)))
                    extra_path.append(segment)
                
                if '/' in path_rule:
                    url = f"https://{domain}{path_rule}"
                else:
                    url = f"https://{domain}/{path_rule}"
                
                if extra_path:
                    url += '/' + '/'.join(extra_path)
            else:
                # Fallback to our original tracker patterns
                tracker_pattern = random.choice(TRACKER_PATTERNS)
                path_depth = random.randint(1, 3)
                path = []
                for _ in range(path_depth):
                    segment = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3,8)))
                    path.append(segment)
                
                url = f"https://{domain}/{'/'.join(path)}"
                
                # Add the tracker pattern
                if tracker_pattern.startswith('/'):
                    url += tracker_pattern
                else:
                    url = url.replace(domain, tracker_pattern)
    else:
        # For non-tracker URLs, generate normally but ensure they don't match EasyPrivacy rules
        while True:
            domain = random.choice(TOP_DOMAINS)
            path_depth = random.randint(1, 4)
            
            # Generate path segments
            path = []
            for _ in range(path_depth):
                segment = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(3,8)))
                path.append(segment)
            
            # Add file extension for static resources
            if random.random() > 0.3:  # 70% chance of static resource
                path[-1] += random.choice(STATIC_EXTS)
            
            # Construct URL
            url = f"https://{domain}/{'/'.join(path)}"
            
            # Add non-tracking parameters occasionally
            if random.random() < 0.3:  # 30% chance of parameters
                params = ['page', 'id', 'lang', 'view', 'q', 'theme']
                query = []
                for _ in range(random.randint(1,2)):
                    key = random.choice(params)
                    val = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=5))
                    query.append(f"{key}={val}")
                
                if query:
                    url += "?" + "&".join(query)
            
            # Check if URL matches any EasyPrivacy rule
            if not any(rule in url for rule in easyprivacy_rules):
                break  # Found a non-tracker URL
    
    return url

def verify_tracker_status(url, easyprivacy_rules):
    """Verify if a URL is a tracker according to EasyPrivacy rules"""
    return any(rule in url for rule in easyprivacy_rules)

def generate_dataset(n=15000, tracker_ratio=0.35):
    """Generate a dataset with exactly tracker_ratio of tracking URLs"""
    data = []
    start_time = datetime.now() - timedelta(days=30)
    
    # Load EasyPrivacy rules
    easyprivacy_rules = load_easyprivacy_rules()
    
    # Calculate exact counts for trackers and non-trackers
    tracker_count = int(n * tracker_ratio)
    non_tracker_count = n - tracker_count
    
    # Generate trackers
    generated_trackers = 0
    while generated_trackers < tracker_count:
        url = generate_realistic_url(True, easyprivacy_rules)
        
        # Verify it's actually a tracker according to EasyPrivacy
        if verify_tracker_status(url, easyprivacy_rules):
            timestamp = start_time + timedelta(seconds=random.randint(0, 2592000))
            data.append({
                "timestamp": timestamp.isoformat(),
                "url": url,
                "status_code": random.choices(
                    [200, 404, 302, 500], 
                    weights=[0.85, 0.05, 0.08, 0.02]
                )[0],
                "content_type": random.choice([
                    "text/html", "application/javascript", 
                    "image/png", "text/css"
                ]),
                "is_tracker": 1
            })
            generated_trackers += 1
            
            if generated_trackers % 500 == 0:
                print(f"Generated {generated_trackers}/{tracker_count} trackers")
    
    # Generate non-trackers
    generated_non_trackers = 0
    while generated_non_trackers < non_tracker_count:
        url = generate_realistic_url(False, easyprivacy_rules)
        
        # Verify it's actually not a tracker according to EasyPrivacy
        if not verify_tracker_status(url, easyprivacy_rules):
            timestamp = start_time + timedelta(seconds=random.randint(0, 2592000))
            data.append({
                "timestamp": timestamp.isoformat(),
                "url": url,
                "status_code": random.choices(
                    [200, 404, 302, 500], 
                    weights=[0.85, 0.05, 0.08, 0.02]
                )[0],
                "content_type": random.choice([
                    "text/html", "application/javascript", 
                    "image/png", "text/css"
                ]),
                "is_tracker": 0
            })
            generated_non_trackers += 1
            
            if generated_non_trackers % 500 == 0:
                print(f"Generated {generated_non_trackers}/{non_tracker_count} non-trackers")
    
    # Shuffle the data to mix trackers and non-trackers
    random.shuffle(data)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate dataset with exactly 35% trackers
    df = generate_dataset(n=1000, tracker_ratio=0.35)
    
    # Verify the distribution
    tracker_count = df['is_tracker'].sum()
    total_count = len(df)
    tracker_percentage = (tracker_count / total_count) * 100
    
    print(f"Generated {len(df)} requests")
    print(f"Tracker distribution: {tracker_count} trackers ({tracker_percentage:.2f}%), {total_count - tracker_count} non-trackers ({100 - tracker_percentage:.2f}%)")
    print("Sample trackers:")
    print(df[df['is_tracker'] == 1].sample(3)['url'].tolist())

    print('synthetic_data_generator part done...')
    
    # Save final dataset (already labeled according to EasyPrivacy)
    training_df = df[['url', 'is_tracker']]
    training_df.to_csv("./data/testing_data_2804.csv", index=False)
    
    print(f"Training dataset created with {len(training_df)} samples")
    print("Training class distribution:\n", training_df["is_tracker"].value_counts())
    print(f"Final tracker percentage: {(training_df['is_tracker'].sum() / len(training_df)) * 100:.2f}%")
