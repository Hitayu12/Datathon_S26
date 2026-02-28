# Datathon_S26

## SignalForge: Failure Intelligence Studio

SignalForge is a Streamlit app that verifies whether a company actually failed, benchmarks survivor peers, runs a digital twin counterfactual, and exports a polished analyst report.

## Input

- Single input field: company full name or ticker.

## What It Delivers

- Failure verification gate (Tavily + Groq)
- Survivor cohort benchmarking
- Digital twin simulation (risk before vs after)
- Robust NLP forensics layer:
  - Negation-aware distress parsing
  - Theme-level severity scoring and evidence extraction
  - Distress intensity/confidence scoring
- Plain-English summary + technical deep-dive
- Trained local analyst model (logistic classifier on distress scenarios)
- Judge-ready features:
  - Interactive Scenario Lab
  - Ask Report Q&A
  - Hover glossary/tooltips
  - JSON + Markdown export

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- API keys are loaded automatically from `.env` (`TAVILY_API_KEY`, `GROQ_API_KEY`).
- IBM watsonx.ai is also supported for enterprise-aligned inference. Set `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, and `WATSONX_MODEL` in `.env`.
- Switch LLM providers at runtime from the Streamlit sidebar using `LLM Provider`.
- watsonx.ai is included for sponsor alignment and a more enterprise-native inference path. It allows the same structured reasoning flow to run on IBM infrastructure without changing the app pipeline.
- If historical filings are sparse or symbol mapping is weak, the app uses transparent metric estimation so risk simulation still works.
