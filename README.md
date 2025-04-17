# llm-anti-tracking
A repo with the code, data and models needed for LLM-based anti-tracking workshop as part of ML Prague 2025. 

## About
### Title: Utilizing Large Language Models for improved anti-tracking in web browsers
### Abstract 
Online tracking remains a significant privacy concern for internet users. Current solutions while effective have limitations in terms of coverage maintenance and precision. This workshop aims to leverage the power of LLMs to create a more robust adaptive and efficient anti-tracking system. We will explore the architecture of an LLM-based anti-tracking system developing the data pipeline and exploring how these models can be fine-tuned to analyze network requests page content and user interactions in real-time. The system's ability to understand the semantic context of web elements allows for more accurate identification of tracking attempts reducing false positives while improving detection rates of sophisticated trackers. A key focus will be on the practical challenges of implementing such a system within the constraints of a web browser environment. We'll discuss strategies for optimizing LLM inference to meet the real-time demands of browsing balancing accuracy with performance.

### Learning Objectives
By the end of this workshop, participants will:
1.	**Understand** how online tracking works and its challenges.
2.	**Learn** how to use LLMs to analyze and classify network requests, page content, and user interactions.
3.	**Build a data pipeline** for collecting and processing web tracking data.
4.	**Fine-tune** an LLM on labeled tracking data.
5.	**Deploy** a model inference system in web browser.
6.	**Evaluate / Test** the model for ML performance and browser integration.
7.	**Learn to Optimize** performance to ensure real-time tracking detection.

_**We will be using Mac / Linux as the primary OS for this workshop.**_

## Preparation for the Workshop (TODO Before Workshop)
Please follow all the instructions below to prepare for the workshop. This includes:
- Getting code and data
- Downloading model
- Setting up your machine

**Please complete all three steps before the workshop.**

### Code and Data
Please clone the repo in your local machine to get the code and data required for this project.
```
git clone https://github.com/humeranoor/llm-anti-tracking
```

### Trained Model
- Download the [original pre-trained model](https://drive.google.com/file/d/1FuDfbfiNawnfvTQJ5MZLdzBh5xGt1Bfq/view?usp=drive_link) and save to the ./original_distilbert folder.

- Download the [final trained model](https://drive.google.com/file/d/1flzzMz2d5JUlrCByjy4bZ20wnpDSiOYw/view?usp=sharing) and save to the ./final_model folder.

Note: Both the models have the same name, so please ensure the correct order of copying.

### Setting up the Machine
1. Create virtual environment:

```
python -m venv tracking-llm-venv
source tracking-llm-venv/bin/activate  # macOS/Linux
```

2. Install Python Packages: 
```
cd llm-anti-tracking/
pip install -r requirements.txt
```
