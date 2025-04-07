# collect_and_label.py
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import pandas as pd
import random

# List of real news sites
NEWS_SITES = [
    "https://www.bbc.com",
    "https://www.reuters.com",
    "https://apnews.com",
    "https://www.theguardian.com",
    "https://www.aljazeera.com"
]

def collect_scripts():
    driver = webdriver.Chrome()
    
    try:
        site = random.choice(NEWS_SITES)
        print(f"Collecting from: {site}")
        driver.get(site)
        
        # Wait for page to load properly
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body')))
        
        # Modern Selenium 4 syntax
        scripts = [s.get_attribute('src') 
                   for s in driver.find_elements(By.TAG_NAME, 'script')]
        iframes = [i.get_attribute('src') 
                   for i in driver.find_elements(By.TAG_NAME, 'iframe')]
        images = [i.get_attribute('src') 
                  for i in driver.find_elements(By.TAG_NAME, 'img')]
        
        all_urls = scripts + iframes + images
        return [url for url in all_urls if url and url.startswith('http')]
        
    except (WebDriverException, TimeoutException) as e:
        print(f"Error collecting from {site}: {str(e)}")
        return []
    finally:
        driver.quit()

# Load EasyPrivacy rules
# Load both general and specific tracking rules
try:
    # General tracking patterns
    with open("easylist/easyprivacy/easyprivacy_general.txt") as f:
        general_rules = [line.strip() for line in f if line.strip() and not line.startswith('!')]
    
    # Specific tracker domains
    with open("easylist/easyprivacy/easyprivacy_specific.txt") as f:
        specific_rules = [line.strip() for line in f if line.strip() and not line.startswith('!')]
    
    easyprivacy_rules = general_rules + specific_rules

except FileNotFoundError:
    print("EasyPrivacy files not found. Did you run:")
    print("git clone --depth=1 https://github.com/easylist/easyprivacy.git")
    easyprivacy_rules = [
        r"google-analytics\.com",
        r"doubleclick\.net",
        r"facebook\.net",
        r"googletagmanager\.com",
        r"/analytics\.js",
        r"/tr/",
        r"/tracking-pixel"
    ]# Collect and label data
real_urls = collect_scripts()

if not real_urls:
    print("Using fallback pre-collected data")
    real_urls = [
        "https://www.googletagmanager.com/gtag/js?id=UA-12345",
        "https://cdn.cookielaw.org/scripttemplates/otSDKStub.js"
    ]

# Label URLs using EasyPrivacy
labels = []
for url in real_urls:
    is_tracker = any(rule in url for rule in easyprivacy_rules)
    labels.append(int(is_tracker))

# Combine with synthetic data
synth_data = pd.read_csv("data/requests.csv")
full_data = pd.DataFrame({"url": real_urls, "is_tracker": labels})
full_data = pd.concat([full_data, synth_data[["url", "is_tracker"]]])  # Added ]

# Save final dataset
full_data.to_csv("training_data.csv", index=False)
print(f"Dataset created with {len(full_data)} samples")
print("Class distribution:\n", full_data["is_tracker"].value_counts())