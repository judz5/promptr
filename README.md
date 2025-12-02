# Promptr : An Automated LLM Testing/Comparison Framework

## What is Promptr?

Promptr sends huggingface prompts to specified API endpoints. You can configure the prompt split (benign/malicious), and add as many endpoints/models as you desire.
The results are then stored in a CSV and parsed to make some insightful graphs for understanding weakpoints in your implementation. 

### Why?

You can compare block rates between Palo Alto Prisma AIRS, AWS Bedrock, or any other AI protection systems you utilize! Simply configure the blocking behavior (WIP), and Promptr can begin tracking effectiveness. 

## Use

Install requirments

```bash
pip install -r requirments.txt
```

run the app

```bash
streamlit run app.py
```

Then simply configure your endpoints in the launched GUI and begin promptring.

## Featrues / Futures

- [x] Customizable Endpoints
- [x] Basic Dataset Configurations
- [ ] Customizable Datasets 
- [x] Radar graph comparison
- [ ] Customizable Graphs/Comparisons
- [ ] Customizable Blocking Detection

