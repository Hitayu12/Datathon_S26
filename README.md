# Mo-Hit_Datathon

## SignalForge: Failure Intelligence Studio

SignalForge is a Streamlit app that verifies whether a company actually failed, benchmarks survivor peers, runs a digital twin counterfactual, and exports a polished analyst report.

## Input

- Single input field: company full name or ticker.

## What It Delivers

- Failure verification gate (Tavily + Groq)
- Survivor cohort benchmarking
- Digital twin simulation (risk before vs after)
- Plain-English summary + technical deep-dive
- Trained local analyst model (logistic classifier on distress scenarios)
- Export buttons:
  - JSON report
  - Markdown report

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- API keys are loaded automatically from `.env` (`TAVILY_API_KEY`, `GROQ_API_KEY`).
- If historical filings are sparse or symbol mapping is weak, the app uses transparent metric estimation so risk simulation still works.
