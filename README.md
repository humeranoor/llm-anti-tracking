# llm-anti-tracking
A repo with the code, data and models needed for LLM-based anti-tracking workshop as part of ML Prague 2025. 

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
