"""SignalForge: Failure intelligence dashboard with survivor benchmarking and simulation."""

from __future__ import annotations

import json
import os
import html
import re
import ast
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from collaborative_reasoning import run_reasoning_council
from data_loader import (
    fetch_company_info,
    fetch_company_profile,
    fetch_financials,
    find_peer_companies,
    resolve_company_input,
)
from financial_analysis import compute_metrics
from groq_client import GroqReasoningClient
from local_reasoner import LocalAnalystModel
from nlp_analysis import qualitative_summary
from risk_model import (
    LayeredAnalysisEngine,
    MultiFactorRiskEngine,
    compare_failure_vs_survivors,
    generate_strategy_recommendations,
    simulate_counterfactual,
)
from tavily_client import TavilyClient
from watsonx_client import WatsonxReasoningClient

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


def _inject_styles() -> None:
    st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
<style>
:root{--bg:#000000;--card:rgba(255,255,255,0.04);--card-border:rgba(255,255,255,0.12);--ink:#ffffff;--muted:#f2f4f8;--dim:#c1c7cd;--accent:#0f62fe;--accent2:#82cfff;--teal:#3ddbd9;--green:#3ddbd9;--red:#ff7eb6;--orange:#be95ff;}
html, body, .stApp, .stMarkdown, .stTextInput, .stSelectbox, .stSlider, .stTabs, .stButton, .stCaption, .stMetric {
    font-family: 'Inter', sans-serif;
}
html,body,.stApp{background:#000000!important;color:var(--ink)!important;}
.stApp{background:#000000!important;}
.block-container{padding-top:0.8rem!important;max-width:1280px!important;}
.stApp p,.stApp span,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stMarkdown,[data-testid="stMarkdownContainer"] p{color:var(--ink)!important;}
.stCaption,[data-testid="stCaptionContainer"] p{color:var(--muted)!important;}


[data-testid="stSidebar"]{background:#111111!important;border-right:1px solid rgba(255,255,255,0.06)!important;}
[data-testid="stSidebar"] .stMarkdown p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:var(--ink)!important;}
[data-testid="stSidebar"] [data-baseweb="select"]{background:rgba(255,255,255,0.06)!important;border-color:rgba(255,255,255,0.12)!important;}
[data-testid="stSidebar"] [data-baseweb="select"] span{color:var(--ink)!important;}
div[data-baseweb="input"],div[data-baseweb="base-input"]{
    background:rgba(255,255,255,0.12)!important;
    border:1px solid rgba(255,255,255,0.34)!important;
    border-radius:10px!important;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,0.05)!important;
}
div[data-baseweb="input"] input,div[data-baseweb="base-input"] input{
    color:var(--ink)!important;
    -webkit-text-fill-color:var(--ink)!important;
    caret-color:var(--accent)!important;
}
div[data-baseweb="input"] input::placeholder,
div[data-baseweb="base-input"] input::placeholder{
    color:#d8dde3!important;
    opacity:1!important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="base-input"]:focus-within{
    border:1px solid rgba(15,98,254,0.92)!important;
    box-shadow:0 0 0 1px rgba(15,98,254,0.35)!important;
}
div[data-testid="stWidgetLabel"] p,div[data-testid="stWidgetLabel"] label,.stTextInput label,.stSlider label,.stSelectbox label,.stTextArea label{color:var(--muted)!important;font-weight:500!important;font-size:0.82rem!important;letter-spacing:0.03em!important;text-transform:uppercase!important;}
div[data-testid="stMetric"]{background:var(--card)!important;border:1px solid var(--card-border)!important;border-radius:14px!important;padding:0.8rem 1rem!important;backdrop-filter:blur(8px);transition:transform 0.18s,border-color 0.18s;}
div[data-testid="stMetric"]:hover{transform:translateY(-2px);border-color:rgba(15,98,254,0.4)!important;}
div[data-testid="stMetricLabel"] p{color:var(--muted)!important;font-size:0.78rem!important;font-weight:600!important;letter-spacing:0.04em;text-transform:uppercase;}
div[data-testid="stMetricValue"]{color:var(--ink)!important;font-weight:800!important;font-size:1.6rem!important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.03)!important;border:1px solid rgba(255,255,255,0.07)!important;border-radius:12px!important;padding:4px!important;gap:2px!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;border:none!important;border-radius:9px!important;color:var(--muted)!important;font-weight:600!important;font-size:0.82rem!important;padding:0.42rem 0.9rem!important;transition:all 0.15s;}
.stTabs [data-baseweb="tab"]:hover{color:var(--ink)!important;background:rgba(255,255,255,0.05)!important;}
.stTabs [aria-selected="true"]{background:rgba(15,98,254,0.22)!important;color:#82cfff!important;border:1px solid rgba(15,98,254,0.35)!important;}
.stTabs [data-baseweb="tab"] p,.stTabs [data-baseweb="tab"] span{color:inherit!important;}
.stButton>button,div[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,#0f62fe 0%,#82cfff 100%)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:0.88rem!important;box-shadow:0 0 18px rgba(15,98,254,0.35)!important;transition:filter 0.18s,transform 0.15s,box-shadow 0.18s!important;}
.stButton>button:hover,div[data-testid="stDownloadButton"]>button:hover{filter:brightness(1.12)!important;transform:translateY(-1px)!important;box-shadow:0 0 26px rgba(15,98,254,0.55)!important;}
.stButton>button[kind="secondary"]{background:rgba(255,255,255,0.07)!important;border:1px solid rgba(255,255,255,0.12)!important;box-shadow:none!important;color:var(--muted)!important;}
.stButton>button span,div[data-testid="stDownloadButton"]>button span{color:inherit!important;font-weight:700!important;}
div[data-testid="stDownloadButton"]{width:100%!important;}
div[data-testid="stDownloadButton"]>button{width:100%!important;}
div[data-testid="stExpander"]{background:var(--card)!important;border:1px solid var(--card-border)!important;border-radius:12px!important;backdrop-filter:blur(6px);}
div[data-testid="stExpander"] details summary p,div[data-testid="stExpander"] details summary span{color:var(--ink)!important;font-weight:600!important;}
div[data-baseweb="slider"] [role="slider"]{background:var(--accent)!important;}
[data-baseweb="select"] [data-baseweb="menu"]{background:#111111!important;border:1px solid rgba(255,255,255,0.1)!important;border-radius:10px!important;}
[data-baseweb="select"] li{color:var(--ink)!important;}
[data-baseweb="select"] li:hover{background:rgba(15,98,254,0.15)!important;}
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,0.08)!important;border-radius:12px!important;overflow:hidden;}
[data-testid="stDataFrame"] th{background:rgba(15,98,254,0.15)!important;color:#82cfff!important;font-weight:700!important;}
[data-testid="stDataFrame"] td{color:var(--ink)!important;background:rgba(255,255,255,0.02)!important;}
.stVegaLiteChart,.vega-embed{background:transparent!important;}
div[data-testid="stAlert"]{border-radius:12px!important;border-left-width:4px!important;}
div[data-testid="stVerticalBlockBorderWrapper"]>div{border:1px solid var(--card-border)!important;border-radius:16px!important;background:var(--card)!important;backdrop-filter:blur(8px);padding:1rem!important;}
.hero{background:#000000;border:1px solid rgba(15,98,254,0.25);border-radius:20px;padding:1.5rem 1.8rem;box-shadow:0 0 60px rgba(15,98,254,0.1),0 20px 40px rgba(0,0,0,0.4);margin-bottom:1.2rem;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:240px;height:240px;background:radial-gradient(circle,rgba(15,98,254,0.2) 0%,transparent 70%);border-radius:50%;}
.hero h1{margin:0;font-size:1.85rem;font-weight:800;color:#fff;letter-spacing:-0.02em;}
.hero .subtitle{margin:0.4rem 0 0;color:#a5b4fc;font-size:0.93rem;}
.step-grid{display:grid;grid-template-columns:repeat(3,minmax(110px,1fr));gap:0.65rem;margin-top:1rem;}
.step-item{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:0.65rem 0.75rem;font-size:0.85rem;line-height:1.4;color:#c6c6c6;}
.step-item b{color:#a5b4fc;display:block;margin-bottom:0.2rem;}
.loop-card{border-radius:18px;padding:1.1rem 1.2rem;position:relative;overflow:hidden;min-height:200px;backdrop-filter:blur(10px);}
.loop-card.danger{background:linear-gradient(135deg,rgba(255,126,182,0.12) 0%,rgba(255,126,182,0.05) 100%);border:1px solid rgba(255,126,182,0.35);box-shadow:0 0 30px rgba(255,126,182,0.08);}
.loop-card.success{background:linear-gradient(135deg,rgba(61,219,217,0.12) 0%,rgba(61,219,217,0.05) 100%);border:1px solid rgba(61,219,217,0.35);box-shadow:0 0 30px rgba(61,219,217,0.08);}
.loop-card.purple{background:linear-gradient(135deg,rgba(130,207,255,0.14) 0%,rgba(15,98,254,0.06) 100%);border:1px solid rgba(130,207,255,0.35);box-shadow:0 0 30px rgba(130,207,255,0.1);}
.loop-card-title{font-size:1.05rem;font-weight:700;color:#fff;margin-bottom:0.8rem;}
.loop-stat{display:flex;justify-content:space-between;align-items:center;padding:0.32rem 0;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.87rem;}
.loop-stat:last-of-type{border-bottom:none;}
.loop-stat-label{color:var(--muted);}
.loop-stat-value{font-weight:700;color:var(--ink);}
.loop-footer{margin-top:0.7rem;font-size:0.76rem;color:var(--dim);}
.badge{display:inline-flex;align-items:center;gap:0.3rem;border-radius:999px;font-size:0.78rem;font-weight:700;padding:0.28rem 0.75rem;border:1px solid;letter-spacing:0.03em;}
.badge-ok{background:rgba(61,219,217,0.15);color:#6ee7b7;border-color:rgba(61,219,217,0.35);}
.badge-failed{background:rgba(255,126,182,0.15);color:#fca5a5;border-color:rgba(255,126,182,0.35);}
.badge-warn{background:rgba(190,149,255,0.15);color:#fcd34d;border-color:rgba(190,149,255,0.35);}
.explain{background:rgba(15,98,254,0.06);border:1px solid rgba(15,98,254,0.2);border-left:4px solid var(--accent);border-radius:14px;padding:1rem 1.1rem;color:var(--ink);line-height:1.65;font-size:0.93rem;}
.wow{background:linear-gradient(135deg,rgba(15,98,254,0.1),rgba(20,184,166,0.08));border:1px solid rgba(15,98,254,0.25);border-radius:14px;padding:0.75rem 0.9rem;font-size:0.88rem;color:#82cfff;}
.panel{background:var(--card);border:1px solid var(--card-border);border-radius:16px;padding:1rem 1.2rem;}
.hint{border-bottom:1px dotted var(--accent);cursor:help;font-weight:600;color:#a5b4fc;}
.st-key-glossary_panel .stButton>button{
    background:rgba(15,98,254,0.12)!important;
    border:1px solid rgba(130,207,255,0.35)!important;
    color:#dbeafe!important;
    box-shadow:none!important;
    min-height:40px!important;
    font-weight:700!important;
}
.st-key-glossary_panel .stButton>button:hover{
    background:rgba(15,98,254,0.2)!important;
    border-color:rgba(130,207,255,0.6)!important;
    transform:none!important;
}
.glossary-card{
    background:rgba(15,98,254,0.07);
    border:1px solid rgba(130,207,255,0.28);
    border-radius:12px;
    padding:0.8rem 0.9rem;
    margin-top:0.55rem;
}
.flow-grid{display:grid;grid-template-columns:repeat(3,minmax(160px,1fr));gap:0.55rem;margin:0.6rem 0;}
.flow-card{background:var(--card);border:1px solid var(--card-border);border-radius:12px;padding:0.65rem 0.75rem;color:var(--ink);font-size:0.86rem;line-height:1.35;}
.flow-card b{color:#a5b4fc;display:block;margin-bottom:0.2rem;}
.council-card{background:linear-gradient(135deg,rgba(15,98,254,0.08) 0%,rgba(130,207,255,0.05) 100%);border:1px solid rgba(15,98,254,0.25);border-radius:16px;padding:1.1rem 1.2rem;margin-bottom:0.8rem;position:relative;}
.council-card::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#0f62fe,#0043ce);border-radius:3px 0 0 3px;}
.council-model-badge{display:inline-flex;align-items:center;gap:0.3rem;background:rgba(15,98,254,0.15);color:#a5b4fc;border:1px solid rgba(15,98,254,0.3);border-radius:999px;padding:0.15rem 0.55rem;font-size:0.72rem;font-weight:700;margin-bottom:0.6rem;}
.council-driver{display:flex;align-items:flex-start;gap:0.5rem;background:rgba(255,126,182,0.05);border:1px solid rgba(255,126,182,0.15);border-radius:10px;padding:0.55rem 0.7rem;margin:0.3rem 0;font-size:0.87rem;color:var(--ink);line-height:1.4;}
.council-strategy{display:flex;align-items:flex-start;gap:0.5rem;background:rgba(61,219,217,0.05);border:1px solid rgba(61,219,217,0.15);border-radius:10px;padding:0.55rem 0.7rem;margin:0.3rem 0;font-size:0.87rem;color:var(--ink);line-height:1.4;}
.council-prevention{display:flex;align-items:flex-start;gap:0.5rem;background:rgba(15,98,254,0.05);border:1px solid rgba(15,98,254,0.15);border-radius:10px;padding:0.55rem 0.7rem;margin:0.3rem 0;font-size:0.87rem;color:var(--ink);line-height:1.4;}
.council-exec-summary{background:linear-gradient(135deg,rgba(130,207,255,0.1),rgba(15,98,254,0.06));border:1px solid rgba(130,207,255,0.3);border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:1.2rem;font-size:0.95rem;line-height:1.65;color:#e0e0e0;}
.council-disagree{background:rgba(190,149,255,0.07);border:1px solid rgba(190,149,255,0.25);border-radius:12px;padding:0.65rem 0.8rem;font-size:0.86rem;color:#fcd34d;margin:0.3rem 0;}
.st-key-jarvis_trigger{position:fixed;right:20px;bottom:20px;width:148px;z-index:10001;}
.st-key-jarvis_trigger>div{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important;}
.st-key-jarvis_trigger .stButton>button{height:48px;border-radius:999px!important;background:linear-gradient(135deg,#0f62fe 0%,#14b8a6 100%)!important;border:none!important;box-shadow:0 4px 20px rgba(15,98,254,0.5)!important;font-size:0.85rem!important;font-weight:700!important;color:#fff!important;white-space:nowrap;}
.st-key-jarvis_trigger .stButton>button:hover{box-shadow:0 0 30px rgba(15,98,254,0.7)!important;transform:translateY(-2px)!important;}
.st-key-jarvis_panel{position:fixed;right:20px;bottom:84px;width:min(480px,95vw);max-height:min(76vh,740px);z-index:10000;background:#0a1020;border:1px solid rgba(15,98,254,0.3);border-radius:22px;box-shadow:0 20px 60px rgba(0,0,0,0.6),0 0 40px rgba(15,98,254,0.1);padding:1rem;backdrop-filter:blur(16px);overflow-x:hidden;overflow-y:auto;}
.st-key-jarvis_panel p,.st-key-jarvis_panel label,.st-key-jarvis_panel span,.st-key-jarvis_panel h1,.st-key-jarvis_panel h2,.st-key-jarvis_panel h3{color:#ffffff!important;}
.st-key-jarvis_panel div[data-baseweb="input"],.st-key-jarvis_panel div[data-baseweb="base-input"]{background:rgba(255,255,255,0.06)!important;border:1px solid rgba(15,98,254,0.3)!important;border-radius:999px!important;min-height:44px!important;}
.st-key-jarvis_panel div[data-baseweb="input"] input,.st-key-jarvis_panel div[data-baseweb="base-input"] input{color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
.st-key-jarvis_panel div[data-baseweb="input"] input::placeholder,.st-key-jarvis_panel div[data-baseweb="base-input"] input::placeholder{color:#d0e2ff!important;opacity:1!important;}
.st-key-jarvis_panel form{border:none!important;padding:0!important;margin:0!important;background:transparent!important;}
.st-key-jarvis_panel .stButton>button{border-radius:999px!important;font-size:0.92rem!important;min-height:44px!important;width:100%!important;}
.jarvis-title{color:#f8fafc!important;font-size:1.4rem;font-weight:800;line-height:1;}
.jarvis-sub{color:#94a3b8!important;font-size:0.78rem;margin-top:0.15rem;}
.chat-bubble-ai{background:rgba(15,98,254,0.1);border:1px solid rgba(15,98,254,0.2);border-radius:14px 14px 14px 4px;padding:0.65rem 0.8rem;color:#e0e0e0;font-size:0.91rem;line-height:1.5;margin-bottom:0.4rem;}
.chat-bubble-user{background:rgba(20,184,166,0.1);border:1px solid rgba(20,184,166,0.2);border-radius:14px 14px 4px 14px;padding:0.6rem 0.75rem;color:#ccfbf1;font-size:0.89rem;line-height:1.45;text-align:right;margin-bottom:0.4rem;}
@media(max-width:640px){.st-key-jarvis_panel{right:10px;bottom:74px;width:calc(100vw - 20px);max-height:62vh;}.st-key-jarvis_trigger{right:10px;bottom:12px;}.flow-grid,.step-grid{grid-template-columns:1fr 1fr;}}
/* ── AI Typing Animation ── */
.typing-dots {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 0.2rem 0;
}
.typing-dots span {
    width: 6px;
    height: 6px;
    background-color: var(--accent);
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}
.typing-dots span:nth-child(1) { animation-delay: -0.32s; }
.typing-dots span:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1); }
}

/* ── Consistent Font for Signal Board & Cards ── */
.council-card, .council-card p, .council-card li, .council-card div {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #c6c6c6 !important;
}
.council-driver, .council-strategy {
    font-size: 0.95rem !important;
    color: #f4f4f4 !important;
    margin-bottom: 0.5rem !important;
    padding-left: 1rem !important;
    border-left: 2px solid rgba(255,255,255,0.1);
}
.council-model-badge {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

/* ── Dropdown / Popover Fixes ── */
div[data-baseweb="popover"], div[data-baseweb="popover"] > div { background: #0a1020 !important; }
ul[role="listbox"], ul[role="listbox"] > li { background: #0a1020 !important; color: #f4f4f4 !important; }
li[role="option"]:hover, li[role="option"][aria-selected="true"] { background: rgba(255,255,255,0.1) !important; }
div[role="listbox"] { background: #0a1020 !important; }



</style>
    

""")


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>SignalForge Failure Intelligence</h1>
          <p>Enter a company name or ticker. We verify failure, compare survivors, and show what would have prevented collapse.</p>
          <div class="step-grid">
            <div class="step-item"><b>1) Verify Failure</b><br/>Checks if the case is truly failed/distressed.</div>
            <div class="step-item"><b>2) Benchmark Survivors</b><br/>Finds peers that survived similar stress.</div>
            <div class="step-item"><b>3) Simulate Prevention</b><br/>Recomputes risk if survivor moves were applied.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_llm_badge(provider_name: str, model_name: str) -> None:
    safe_provider = html.escape(provider_name)
    safe_model = html.escape(model_name)
    st.markdown(
        (
            "<div style='margin:0.3rem 0 0.85rem;'>"
            f"<span class='badge badge-ok'>LLM: {safe_provider} | {safe_model}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_council_evidence_bundle(intelligence: Dict[str, object]) -> Dict[str, object]:
    snippets: List[Dict[str, object]] = []
    next_id = 1

    def add_items(label: str, texts: List[object], sources: List[str]) -> None:
        nonlocal next_id
        source_list = list(sources or [])
        for idx, text in enumerate(texts or []):
            cleaned = str(text or "").strip()
            if not cleaned:
                continue
            source = source_list[idx] if idx < len(source_list) else ""
            snippets.append({"id": next_id, "label": label, "text": cleaned, "source": source})
            next_id += 1

    source_groups = dict(intelligence.get("source_groups", {}) or {})
    failure_check = dict(intelligence.get("failure_check", {}) or {})
    add_items("failure_check", [failure_check.get("answer", "")] + list(failure_check.get("snippets", []) or []), list(failure_check.get("sources", []) or []))
    add_items("macro", list(intelligence.get("macro_notes", []) or []), list(source_groups.get("macro", []) or []))
    add_items("micro", list(intelligence.get("micro_notes", []) or []), list(source_groups.get("micro", []) or []))
    add_items("industry", list(intelligence.get("industry_notes", []) or []), list(source_groups.get("industry", []) or []))
    add_items("news", list(intelligence.get("news_notes", []) or []), list(source_groups.get("news", []) or []))
    add_items("strategy", list(intelligence.get("strategy_notes", []) or []), list(source_groups.get("strategy", []) or []))
    add_items("qualitative", list(intelligence.get("qual_snippets", []) or []), list(source_groups.get("qualitative", []) or []))
    return {"snippets": snippets}


def _legacy_reasoning_from_council(council_output: Dict[str, Any]) -> Dict[str, Any]:
    failure_drivers = [str((item or {}).get("driver", "Evidence unavailable")).strip() for item in list(council_output.get("failure_drivers", []) or [])]
    survivor_strategies = [str((item or {}).get("strategy", "Evidence unavailable")).strip() for item in list(council_output.get("survivor_strategies", []) or [])]
    final_recommendations = [str((item or {}).get("action", "Evidence unavailable")).strip() for item in list(council_output.get("final_recommendations", []) or [])]
    technical_notes = [f"Overall council confidence: {float(council_output.get('overall_confidence', 0.0))*100:.1f}%"]
    for row in list(council_output.get("disagreements", []) or [])[:3]:
        topic = str((row or {}).get("topic", "")).strip()
        if topic:
            technical_notes.append(f"Disagreement: {topic}")
    return {
        "plain_english_explainer": str(council_output.get("executive_summary", "Evidence unavailable")),
        "executive_summary": str(council_output.get("executive_summary", "Evidence unavailable")),
        "failure_drivers": failure_drivers[:4] or ["Evidence unavailable"],
        "survivor_differences": survivor_strategies[:4] or ["Evidence unavailable"],
        "prevention_measures": final_recommendations[:4] or ["Evidence unavailable"],
        "technical_notes": technical_notes[:5],
        "model_used": "council",
    }


def _render_council_tab(council_output: Dict[str, Any]) -> None:
    exec_summary = str(council_output.get("executive_summary", "") or "").strip()
    if exec_summary:
        st.markdown(
            f"<div class='council-exec-summary'>🧠 <strong>Executive Summary</strong><br/><br/>{html.escape(exec_summary)}</div>",
            unsafe_allow_html=True,
        )

    # Metrics row
    impact = dict(council_output.get("counterfactual_impact", {}) or {})
    confidence = float(council_output.get("overall_confidence", 0.0))
    before_score = float(impact.get("before_score", 0.0))
    after_score = float(impact.get("after_score", 0.0))
    reduction = max(0.0, before_score - after_score)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Council Confidence", f"{confidence*100:.1f}%")
    c2.metric("Risk Before", f"{before_score:.2f}")
    c3.metric("Risk After Fixes", f"{after_score:.2f}")
    c4.metric("Risk Reduction", f"{reduction:.2f}", delta=f"-{reduction:.2f}" if reduction > 0 else "No change")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        drivers = list(council_output.get("failure_drivers", []) or [])
        if drivers:
            items_html = ""
            for item in drivers[:6]:
                d = html.escape(_coerce_claim_text(item, ["driver", "strategy", "action"]) or "Unknown driver")
                conf_value = _coerce_claim_confidence(item)
                conf_html = (
                    f"<span style='color:#ff7eb6;font-weight:700;font-size:0.78rem;margin-left:auto;white-space:nowrap;'>{conf_value*100:.0f}% conf</span>"
                    if conf_value is not None
                    else ""
                )
                items_html += f"<div class='council-driver'><span>{d} {conf_html}</span></div>"
            st.markdown(
                f"<div class='council-card'>"
                f"<div class='council-model-badge'>⚡ Failure Drivers</div>"
                f"{items_html}</div>",
                unsafe_allow_html=True,
            )

        recommendations = list(council_output.get("final_recommendations", []) or [])
        if recommendations:
            items_html = ""
            for i, item in enumerate(recommendations[:5], 1):
                action = html.escape(_coerce_claim_text(item, ["action", "strategy", "driver"]))
                effect = html.escape(str((item or {}).get("expected_effect", "")) if isinstance(item, dict) else "")
                conf_value = _coerce_claim_confidence(item)
                conf_html = (
                    f"<span style='color:#a5b4fc;font-size:0.76rem;'> — {conf_value*100:.0f}% conf</span>"
                    if conf_value is not None
                    else ""
                )
                items_html += (
                    f"<div class='council-prevention'>"
                    f"<span><strong>{i}. {action}</strong>"
                    f"{'<br/><span style=color:var(--muted);font-size:0.82rem>' + effect + '</span>' if effect else ''}"
                    f"{conf_html}</span></div>"
                )
            st.markdown(
                f"<div class='council-card'>"
                f"<div class='council-model-badge'>🛡 Final Recommendations</div>"
                f"{items_html}</div>",
                unsafe_allow_html=True,
            )

    with col_b:
        strategies = list(council_output.get("survivor_strategies", []) or [])
        if strategies:
            items_html = ""
            for item in strategies[:6]:
                s = html.escape(_coerce_claim_text(item, ["strategy", "action", "driver"]) or "Unknown strategy")
                conf_value = _coerce_claim_confidence(item)
                conf_html = (
                    f"<span style='color:#3ddbd9;font-weight:700;font-size:0.78rem;'>{conf_value*100:.0f}%</span>"
                    if conf_value is not None
                    else ""
                )
                items_html += f"<div class='council-strategy'><span>{s} {conf_html}</span></div>"
            st.markdown(
                f"<div class='council-card'>"
                f"<div class='council-model-badge' style='background:rgba(61,219,217,0.15);color:#6ee7b7;border-color:rgba(61,219,217,0.3);'>✓ Survivor Strategies</div>"
                f"{items_html}</div>",
                unsafe_allow_html=True,
            )

        disagreements = list(council_output.get("disagreements", []) or [])
        if disagreements:
            items_html = ""
            for row in disagreements[:4]:
                topic = html.escape(str((row or {}).get("topic", "Open issue")))
                groq_v = html.escape(str((row or {}).get("groq_view", "")))
                wx_v = html.escape(str((row or {}).get("watsonx_view", "")))
                items_html += (
                    f"<div class='council-disagree'><strong>⚠ {topic}</strong>"
                    f"{'<br/><span style=color:var(--muted)>Groq: ' + groq_v + '</span>' if groq_v else ''}"
                    f"{'<br/><span style=color:var(--muted)>watsonx: ' + wx_v + '</span>' if wx_v else ''}"
                    f"</div>"
                )
            st.markdown(
                f"<div class='council-card' style='border-color:rgba(190,149,255,0.3)'>"
                f"<div class='council-model-badge' style='background:rgba(190,149,255,0.15);color:#fcd34d;border-color:rgba(190,149,255,0.3);'>⚠ Council Disagreements</div>"
                f"{items_html}</div>",
                unsafe_allow_html=True,
            )

    # Model breakdown as compact expander
    breakdown = dict(council_output.get("model_breakdown", {}) or {})
    with st.expander("🔍 Model Breakdown (Groq / watsonx / Local)", expanded=False):
        for key, label in [("groq", "Groq LLM"), ("watsonx", "IBM watsonx.ai"), ("local", "Local NLP/Analyst")]:
            row = dict(breakdown.get(key, {}) or {})
            latency = int(row.get("latency_ms", 0) or 0)
            provider_name = "IBM watsonx.ai" if key == "watsonx" else ("Groq" if key == "groq" else "Local")
            raw_error = row.get("errors")
            errors = _normalize_provider_error(provider_name, raw_error) if raw_error else "None"
            signal_summary = dict(row.get("signal_summary", {}) or {})
            snippet_count = int(signal_summary.get("snippet_count", 0) or 0)
            source_count = int(signal_summary.get("source_count", 0) or 0)
            channels = ", ".join(list(signal_summary.get("channels", []) or [])[:5]) or "n/a"
            st.markdown(
                f"<div class='council-model-badge'>🤖 {label}</div> "
                f"<span style='color:var(--muted);font-size:0.82rem;'>"
                f"Latency: {latency}ms | Signals: {snippet_count} snippets ({source_count} sourced) | "
                f"Channels: {html.escape(channels)} | Errors: {html.escape(errors)}</span>",
                unsafe_allow_html=True,
            )




def _render_council_trace_tab(council_output: Dict[str, Any], qual: Dict[str, Any] | None = None) -> None:
    """Show how the Collaborative Reasoning Council formed its conclusions, including NLP input."""
    import plotly.graph_objects as go

    # ── Pipeline diagram ──────────────────────────────────────────────────
    nlp_intensity = float((qual or {}).get("distress_intensity", 0.0)) if qual else 0.0
    nlp_themes = list((qual or {}).get("theme_signals", [])) if qual else []
    nlp_conf = float((qual or {}).get("confidence", 0.0)) if qual else 0.0
    nlp_summary = str((qual or {}).get("forensic_summary", "")) if qual else ""

    st.markdown(
        f"""
        <div class="flow-grid" style="grid-template-columns:repeat(5,1fr);">
          <div class="flow-card" style="border-color:rgba(20,184,166,0.4);background:rgba(20,184,166,0.06);">
            <b style="color:#14b8a6;">① NLP Engine</b><br/>
            Scans financial text + synth metrics. Intensity: <b style="color:#f4f4f4;">{nlp_intensity:.1f}/10</b><br/>
            Themes: <em style="color:#6ee7b7;">{', '.join(nlp_themes[:3]) or 'none detected'}</em>
          </div>
          <div class="flow-card" style="border-color:rgba(15,98,254,0.4);background:rgba(15,98,254,0.06);">
            <b style="color:#a5b4fc;">② Groq Draft</b><br/>
            Generates structured failure narrative from metrics, peers, NLP signals, and Tavily evidence.
          </div>
          <div class="flow-card" style="border-color:rgba(130,207,255,0.4);background:rgba(130,207,255,0.06);">
            <b style="color:#82cfff;">③ watsonx Critique</b><br/>
            Challenges weakly-supported claims and forces citations back to evidence IDs.
          </div>
          <div class="flow-card" style="border-color:rgba(190,149,255,0.4);background:rgba(190,149,255,0.06);">
            <b style="color:#fcd34d;">④ Local Sanity</b><br/>
            Checks narrative consistency against quantitative risk score + NLP forensic summary.
          </div>
          <div class="flow-card" style="border-color:rgba(61,219,217,0.4);background:rgba(61,219,217,0.06);">
            <b style="color:#6ee7b7;">⑤ Synthesis</b><br/>
            Consensus keeps high-conf claims, downgrades weak ones, records disagreements.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── NLP Contribution Panel ────────────────────────────────────────────
    if qual and nlp_intensity > 0:
        theme_scores = dict((qual or {}).get("theme_scores", {}) or {})
        top_themes = sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
        theme_html = "".join(
            f"<div class='loop-stat'><span class='loop-stat-label'>{_friendly_theme_name(t)}</span>"
            f"<span class='loop-stat-value' style='color:{'#ff7eb6' if s > 0.4 else '#f97316' if s > 0.2 else '#94a3b8'};'>"
            f"{'🔴' if s > 0.4 else '🟠' if s > 0.2 else '⚪'} {s:.2f}</span></div>"
            for t, s in top_themes
        )
        st.markdown(
            f"<div class='council-card' style='border-color:rgba(20,184,166,0.35);'>"
            f"<div class='council-model-badge' style='background:rgba(20,184,166,0.15);color:#6ee7b7;border-color:rgba(20,184,166,0.3);'>"
            f"🔬 NLP Contribution to Council — Intensity {nlp_intensity:.1f}/10 • Confidence {nlp_conf*100:.0f}%</div>"
            f"<p style='color:#94a3b8;font-size:0.85rem;margin:0.4rem 0;'>{html.escape(nlp_summary)}</p>"
            f"{theme_html}</div>",
            unsafe_allow_html=True,
        )

    # ── Latency chart (Plotly) ────────────────────────────────────────────
    breakdown = dict(council_output.get("model_breakdown", {}) or {})
    watson_row = dict(breakdown.get("watsonx", {}) or {})
    watson_error = str(watson_row.get("errors") or "").strip()
    if watson_error:
        status_msg = "watsonx unavailable this run; Groq+Local kept the council running."
        if _is_watson_quota_error(watson_error):
            status_msg = "watsonx quota/permission is currently blocked; Groq+Local continued using the same sourced evidence."
        st.info(status_msg)

    systems = ["Groq LLM", "IBM watsonx.ai", "Local NLP/Analyst"]
    latencies = [
        int((breakdown.get("groq", {}) or {}).get("latency_ms", 0) or 0),
        int((breakdown.get("watsonx", {}) or {}).get("latency_ms", 0) or 0),
        int((breakdown.get("local", {}) or {}).get("latency_ms", 0) or 0),
    ]
    statuses = [
        "Error" if (breakdown.get(k, {}) or {}).get("errors") else "OK"
        for k in ["groq", "watsonx", "local"]
    ]
    bar_colors = ["#ff7eb6" if s == "Error" else "#0f62fe" for s in statuses]

    if any(latencies):
        fig_lat = go.Figure(go.Bar(
            x=systems, y=latencies, marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v}ms" for v in latencies], textposition="outside",
            textfont=dict(color="#f4f4f4", family="Inter"),
            hovertemplate="<b>%{x}</b><br>Latency: %{y}ms<extra></extra>",
        ))
        fig_lat.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=0, r=0, t=10, b=0), height=220,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", title="ms"),
            title=dict(text="System Latency", font=dict(color="#94a3b8", size=13)),
        )
        c_left, c_right = st.columns([2, 1])
        c_left.plotly_chart(fig_lat, use_container_width=True, config={"displayModeBar": False})
        with c_right:
            st.metric("Council Confidence", f"{float(council_output.get('overall_confidence', 0.0))*100:.1f}%")
            st.metric("Disagreements", f"{len(list(council_output.get('disagreements', []) or []))}")
            total_ms = sum(latencies)
            st.metric("Total Pipeline Time", f"{total_ms}ms")

    # ── Per-system readable outputs ───────────────────────────────────────
    for key, label, accent in [("groq", "Groq LLM Draft", "#0f62fe"), ("watsonx", "watsonx Critique", "#0043ce"), ("local", "Local Sanity Check", "#14b8a6")]:
        row = dict(breakdown.get(key, {}) or {})
        raw = dict(row.get("raw", {}) or {})
        provider_name = "IBM watsonx.ai" if key == "watsonx" else ("Groq" if key == "groq" else "Local")
        raw_error = row.get("errors")
        errors = _normalize_provider_error(provider_name, raw_error) if raw_error else ""
        signal_summary = dict(row.get("signal_summary", {}) or {})
        with st.expander(f"{label} — {'⚠ Error' if errors else '✓ OK'} ({int(row.get('latency_ms', 0) or 0)}ms)", expanded=False):
            if errors:
                st.markdown(f"<div class='council-disagree'>Error: {html.escape(str(errors))}</div>", unsafe_allow_html=True)
            st.caption(
                f"Signals used: {int(signal_summary.get('snippet_count', 0) or 0)} snippets | "
                f"Sourced links: {int(signal_summary.get('source_count', 0) or 0)} | "
                f"Channels: {', '.join(list(signal_summary.get('channels', []) or [])[:6]) or 'n/a'}"
            )
            if raw:
                # Show readable fields instead of raw JSON
                for field_key in ["executive_summary", "failure_drivers", "survivor_strategies",
                                  "narrative_alignment_flags", "failures_confirmed", "critique_notes"]:
                    val = raw.get(field_key)
                    if val:
                        st.markdown(f"**{field_key.replace('_', ' ').title()}**")
                        if isinstance(val, list):
                            for item in val[:5]:
                                if isinstance(item, dict):
                                    item_text = item.get("driver") or item.get("strategy") or item.get("flag") or str(item)
                                    st.write(f"- {item_text}")
                                else:
                                    st.write(f"- {item}")
                        else:
                            st.markdown(f"<div class='explain' style='font-size:0.86rem;'>{html.escape(str(val))}</div>", unsafe_allow_html=True)

    # ── Disagreement table ────────────────────────────────────────────────
    disagreements = list(council_output.get("disagreements", []) or [])
    if disagreements:
        st.markdown("#### 🔀 Where Systems Disagreed")
        diff_rows = [
            {"Topic": r.get("topic", ""), "Groq": r.get("groq_view", ""), "watsonx": r.get("watsonx_view", ""), "Local": r.get("local_view", "")}
            for r in disagreements[:8]
        ]
        st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)





def _inject_assistant_panel_mode_style(compact: bool) -> None:
    if compact:
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
<style>
            .st-key-jarvis_panel {
                width: min(340px, 90vw) !important;
                max-height: min(56vh, 460px) !important;
            }
            .st-key-jarvis_panel .jarvis-title {
                font-size: 1.25rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet" />
<style>
        .st-key-jarvis_panel {
            width: min(470px, 95vw) !important;
            max-height: min(74vh, 720px) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _test_llm_connection(reasoning_client: object, provider_name: str) -> Tuple[bool, str]:
    system_prompt = 'Return JSON only with key "ok" and boolean value true.'
    user_prompt = "Respond with the requested JSON."

    if provider_name == "Groq":
        response = reasoning_client._chat_json(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_completion_tokens=24,
        )
        if not response:
            detail = getattr(reasoning_client, "last_error", "") or "No response returned."
            raise RuntimeError(detail)
        return True, f"Connected. Model: {response.get('model_used', 'unknown')}"

    response = reasoning_client._chat_json(
        system_prompt,
        user_prompt,
        temperature=0.0,
        max_tokens=24,
    )
    return True, f"Connected. Model: {response.get('model_used', 'unknown')}"


def _enabled_provider_chain(
    *,
    provider_choice: str,
    groq_client: GroqReasoningClient,
    watsonx_client: Optional[WatsonxReasoningClient],
) -> List[Tuple[str, object]]:
    ordered: List[Tuple[str, object]] = []
    if provider_choice == "IBM watsonx.ai":
        ordered.append(("IBM watsonx.ai", watsonx_client))
        ordered.append(("Groq", groq_client))
    else:
        ordered.append(("Groq", groq_client))
        ordered.append(("IBM watsonx.ai", watsonx_client))

    chain: List[Tuple[str, object]] = []
    for name, client in ordered:
        if client is None:
            continue
        if name == "Groq" and not bool(getattr(client, "enabled", False)):
            continue
        chain.append((name, client))
    return chain


def _normalize_provider_error(provider_name: str, message: object) -> str:
    raw = str(message or "").strip()
    if not raw:
        return "Unknown provider error."
    if provider_name == "IBM watsonx.ai":
        return WatsonxReasoningClient.summarize_error(raw)
    compact = " ".join(raw.split())
    return compact if len(compact) <= 260 else f"{compact[:257]}..."


def _is_watson_quota_error(message: object) -> bool:
    return WatsonxReasoningClient.is_quota_error(str(message or ""))


def _invoke_with_provider_failover(
    *,
    provider_chain: List[Tuple[str, object]],
    method_name: str,
    payload: Dict[str, Any],
    validator: Any,
) -> Tuple[Optional[Dict[str, Any]], List[str], str]:
    errors: List[str] = []
    primary_name = provider_chain[0][0] if provider_chain else "None"
    for provider_name, client in provider_chain:
        fn = getattr(client, method_name, None)
        if not callable(fn):
            errors.append(f"{provider_name}: method `{method_name}` not available.")
            continue
        try:
            result = fn(**payload)
        except Exception as exc:
            normalized = _normalize_provider_error(provider_name, exc)
            errors.append(f"{provider_name}: {type(exc).__name__}: {normalized}")
            continue
        if isinstance(result, dict) and bool(validator(result)):
            out = dict(result)
            out.setdefault("provider_used", provider_name)
            if provider_name != primary_name and errors:
                out["provider_failover"] = f"Primary provider unavailable. Used {provider_name}."
                out["provider_errors"] = errors[:3]
            return out, errors, provider_name
        errors.append(f"{provider_name}: invalid response payload.")
    return None, errors, primary_name


def _fallback_failure_status(errors: List[str]) -> Dict[str, Any]:
    note = " / ".join(errors[:2]) if errors else "No provider response."
    return {
        "is_failed": False,
        "status_label": "unclear",
        "confidence": 0.5,
        "reason": f"Provider fallback triggered. {note}",
        "evidence": [],
        "model_used": "fallback",
        "provider_used": "fallback",
    }


def _fallback_reasoning(recommendations: List[str], errors: List[str]) -> Dict[str, Any]:
    note = " / ".join(errors[:2]) if errors else "No provider response."
    return {
        "plain_english_explainer": "The model providers were unavailable, so this summary uses deterministic fallback logic.",
        "executive_summary": f"Provider fallback triggered. {note}",
        "failure_drivers": ["Provider calls failed; relying on metric-driven fallback summary."],
        "survivor_differences": ["Survivors generally maintain stronger liquidity and lower leverage."],
        "prevention_measures": recommendations[:3] or ["Stabilize liquidity and reduce leverage."],
        "technical_notes": [f"Provider errors: {note}"],
        "model_used": "fallback",
        "provider_used": "fallback",
    }


def _fallback_answer(errors: List[str]) -> Dict[str, Any]:
    note = " / ".join(errors[:2]) if errors else "No provider response."
    return {
        "answer": "I could not generate a model-backed answer right now.",
        "rationale": f"Provider fallback triggered. {note}",
        "caveat": "Please retry after checking provider connectivity.",
        "confidence": "low",
        "provider_used": "fallback",
    }


def _fmt_num(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{n:.3f}" if abs(n) < 10 else f"{n:.2f}"


def _friendly_metric_name(name: str) -> str:
    mapping = {
        "debt_to_equity": "Debt / Equity",
        "current_ratio": "Current Ratio",
        "cash_burn": "Cash Burn",
        "revenue_growth": "Revenue Growth",
        "expense_growth": "Expense Growth",
        "inventory_growth": "Inventory Growth",
        "gross_margin": "Gross Margin",
        "operating_margin": "Operating Margin",
        "total_debt": "Total Debt",
        "cash_and_equivalents": "Cash & Equivalents",
        "revenue": "Revenue",
        "operating_expense": "Operating Expense",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _friendly_theme_name(name: str) -> str:
    mapping = {
        "liquidity_concerns": "Liquidity Concerns",
        "debt_stress": "Debt Stress",
        "demand_decline": "Demand Decline",
        "margin_pressure": "Margin Pressure",
        "legal_regulatory": "Legal / Regulatory",
        "bankruptcy_language": "Bankruptcy Language",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _clean_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            out[k] = round(float(v), 6)
        else:
            out[k] = v
    return out


def _format_signal_items(notes: List[object], *, max_items: int = 3) -> List[str]:
    formatted: List[str] = []
    for raw in list(notes or []):
        text = " ".join(str(raw or "").split()).strip()
        if not text:
            continue
        text = re.sub(r"^[\-\*\u2022\.\:;\s]+", "", text).strip()
        text = re.sub(r"^(macro|micro|industry|news)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
        if text and text[-1] not in ".!?":
            text = f"{text}."
        formatted.append(text)
        if len(formatted) >= max_items:
            break

    if not formatted:
        return ["Signal 1: No strong signal captured from current sources."]
    return [f"Signal {idx + 1}: {line}" for idx, line in enumerate(formatted)]


def _layer_context_details(
    *,
    layer_key: str,
    intelligence: Dict[str, object],
    qual: Dict[str, object],
    failing_metrics: Dict[str, Optional[float]],
) -> Tuple[List[str], List[str], str]:
    used_by = {
        "macro": "Risk score (macro component) + layered diagnostics + council reasoning",
        "business_model": "Layered diagnostics + council reasoning",
        "financial_health": "Risk score core components + layered diagnostics + council reasoning",
        "operational": "Layered diagnostics + council reasoning",
        "qualitative": "Local NLP distress model + layered diagnostics + council reasoning",
    }

    if layer_key == "macro":
        channels = ["Macro", "News"]
        evidence = _format_signal_items(
            list(intelligence.get("macro_notes", []) or []) + list(intelligence.get("news_notes", []) or []),
            max_items=2,
        )
        return channels, evidence, used_by[layer_key]

    if layer_key == "business_model":
        channels = ["Micro", "Industry", "News"]
        evidence = _format_signal_items(
            list(intelligence.get("micro_notes", []) or []) + list(intelligence.get("industry_notes", []) or []),
            max_items=2,
        )
        return channels, evidence, used_by[layer_key]

    if layer_key == "financial_health":
        channels = ["Financial Metrics", "Micro"]
        evidence: List[str] = []
        dte = failing_metrics.get("debt_to_equity")
        cr = failing_metrics.get("current_ratio")
        burn = failing_metrics.get("cash_burn")
        rev = failing_metrics.get("revenue")
        if dte is not None:
            evidence.append(f"Signal 1: Debt / Equity observed at {float(dte):.2f}.")
        if cr is not None:
            evidence.append(f"Signal {len(evidence)+1}: Current ratio observed at {float(cr):.2f}.")
        if burn is not None and rev not in (None, 0):
            burn_ratio = abs(float(burn)) / max(abs(float(rev)), 1.0)
            evidence.append(f"Signal {len(evidence)+1}: Cash burn intensity at {burn_ratio*100:.1f}% of revenue.")
        if not evidence:
            evidence = _format_signal_items(list(intelligence.get("micro_notes", []) or []), max_items=2)
        return channels, evidence[:2], used_by[layer_key]

    if layer_key == "operational":
        channels = ["Financial Metrics", "Industry", "News"]
        evidence = []
        exp_growth = failing_metrics.get("expense_growth")
        rev_growth = failing_metrics.get("revenue_growth")
        op_margin = failing_metrics.get("operating_margin")
        inv_growth = failing_metrics.get("inventory_growth")
        if exp_growth is not None and rev_growth is not None:
            evidence.append(
                f"Signal 1: Expense growth ({float(exp_growth):.2f}) vs revenue growth ({float(rev_growth):.2f})."
            )
        if op_margin is not None:
            evidence.append(f"Signal {len(evidence)+1}: Operating margin observed at {float(op_margin):.2f}.")
        if inv_growth is not None:
            evidence.append(f"Signal {len(evidence)+1}: Inventory growth observed at {float(inv_growth):.2f}.")
        if not evidence:
            evidence = _format_signal_items(
                list(intelligence.get("industry_notes", []) or []) + list(intelligence.get("news_notes", []) or []),
                max_items=2,
            )
        return channels, evidence[:2], used_by[layer_key]

    # qualitative
    channels = ["Qualitative NLP", "Micro", "News"]
    theme_evidence = dict(qual.get("theme_evidence", {}) or {})
    snippets: List[str] = []
    for _, values in theme_evidence.items():
        for item in list(values or [])[:1]:
            text = " ".join(str(item or "").split()).strip()
            if text:
                snippets.append(text)
        if len(snippets) >= 2:
            break
    if snippets:
        evidence = [f"Signal {idx+1}: {txt if txt.endswith(('.', '!', '?')) else txt + '.'}" for idx, txt in enumerate(snippets[:2])]
    else:
        evidence = _format_signal_items(
            list(intelligence.get("news_notes", []) or []) + list(intelligence.get("micro_notes", []) or []),
            max_items=2,
        )
    return channels, evidence[:2], used_by["qualitative"]


def _coerce_claim_text(item: Any, preferred_keys: List[str]) -> str:
    payload: Dict[str, Any] = {}
    if isinstance(item, dict):
        payload = item
    elif isinstance(item, str):
        raw = item.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    payload = parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}
        if not payload:
            return raw

    if payload:
        for key in preferred_keys:
            candidate = str(payload.get(key, "")).strip()
            if candidate:
                return candidate
        return str(payload).strip()

    return str(item).strip()


def _coerce_claim_confidence(item: Any) -> Optional[float]:
    if isinstance(item, dict):
        raw = item.get("confidence")
        try:
            conf = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, conf))

    if isinstance(item, str):
        raw = item.strip()
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = json.loads(raw)
            except Exception:
                try:
                    parsed = ast.literal_eval(raw)
                except Exception:
                    return None
            if isinstance(parsed, dict):
                try:
                    conf = float(parsed.get("confidence"))
                except (TypeError, ValueError):
                    return None
                return max(0.0, min(1.0, conf))
    return None


def _metrics_table(metrics: Dict[str, object], hide_missing: bool = False) -> pd.DataFrame:
    rows = []
    for k, v in metrics.items():
        if hide_missing and v is None:
            continue
        rows.append({"Metric": _friendly_metric_name(k), "Value": _fmt_num(v)})
    return pd.DataFrame(rows)


def _aligned_metric_tables(
    failing_metrics: Dict[str, object],
    survivor_metrics: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Keep a strict 1:1 comparison: only metrics present (non-null) for failing company are shown.
    ordered_keys = [k for k, v in failing_metrics.items() if v is not None]
    fail_rows: List[Dict[str, str]] = []
    surv_rows: List[Dict[str, str]] = []
    for key in ordered_keys:
        fail_rows.append({"Metric": _friendly_metric_name(key), "Value": _fmt_num(failing_metrics.get(key))})
        surv_rows.append({"Metric": _friendly_metric_name(key), "Value": _fmt_num(survivor_metrics.get(key))})
    return pd.DataFrame(fail_rows), pd.DataFrame(surv_rows)


def _core_metric_quality(metrics: Dict[str, Optional[float]]) -> int:
    keys = ["debt_to_equity", "current_ratio", "cash_burn", "revenue_growth", "revenue"]
    return sum(1 for k in keys if metrics.get(k) is not None)


def _looks_distressed_symbol(ticker: str) -> bool:
    t = str(ticker or "").upper().strip()
    return bool(t) and (t.endswith("Q") or t.endswith(".PK"))


def _impute_failed_defaults(metrics: Dict[str, Optional[float]]) -> Tuple[Dict[str, Optional[float]], bool]:
    patched = dict(metrics)
    changed = False

    if patched.get("revenue") is None:
        patched["revenue"] = 1_000_000_000.0
        changed = True
    if patched.get("debt_to_equity") is None:
        patched["debt_to_equity"] = 3.1
        changed = True
    if patched.get("current_ratio") is None:
        patched["current_ratio"] = 0.72
        changed = True
    if patched.get("cash_burn") is None:
        patched["cash_burn"] = 240_000_000.0
        changed = True
    if patched.get("revenue_growth") is None:
        patched["revenue_growth"] = -0.16
        changed = True

    return patched, changed


def _impute_survivor_defaults(metrics: Dict[str, Optional[float]], ticker: str) -> Tuple[Dict[str, Optional[float]], bool]:
    patched = dict(metrics)
    changed = False
    seed = sum(ord(c) for c in ticker)

    if patched.get("revenue") is None:
        patched["revenue"] = float(10_000_000_000 + (seed % 18) * 1_500_000_000)
        changed = True
    if patched.get("debt_to_equity") is None:
        patched["debt_to_equity"] = round(0.8 + (seed % 12) / 20.0, 3)
        changed = True
    if patched.get("current_ratio") is None:
        patched["current_ratio"] = round(1.25 + (seed % 8) / 20.0, 3)
        changed = True
    if patched.get("cash_burn") is None:
        patched["cash_burn"] = 0.0
        changed = True
    if patched.get("revenue_growth") is None:
        patched["revenue_growth"] = round(0.03 + (seed % 6) / 100.0, 3)
        changed = True

    return patched, changed


def _fetch_tavily_intelligence(tavily_client: TavilyClient, company_name: str, ticker: str, industry: str) -> Dict[str, object]:
    if not tavily_client.enabled:
        return {
            "macro_stress_score": 50.0,
            "macro_notes": ["Macro intelligence unavailable (no Tavily key)."],
            "micro_notes": ["Micro/company intelligence unavailable (no Tavily key)."],
            "industry_notes": ["Industry intelligence unavailable (no Tavily key)."],
            "news_notes": ["News intelligence unavailable (no Tavily key)."],
            "qual_snippets": [],
            "sources": [],
            "source_groups": {
                "macro": [],
                "qualitative": [],
                "strategy": [],
                "failure_check": [],
                "micro": [],
                "industry": [],
                "news": [],
            },
            "strategy_notes": [],
            "failure_check": {"answer": "", "snippets": [], "sources": []},
        }

    macro_query = f"Macro stress for {industry} with rates, credit, demand and default pressure"
    qual_query = f"{company_name} {ticker} liquidity risk covenant breach restructuring distress signals"
    strategy_query = f"Survivor strategies for stressed {industry} companies that avoided collapse"
    failure_query = f"Did {company_name} {ticker} fail: chapter 11 bankruptcy liquidation insolvency collapse"
    micro_query = f"{company_name} {ticker} management execution leverage liquidity funding governance breakdown"
    industry_query = f"{industry} industry structure regulation competitive pressure and consolidation risk"
    news_query = f"{company_name} {ticker} timeline of events and major news before distress or failure"

    macro_result = tavily_client.search(macro_query, max_results=4)
    qual_result = tavily_client.search(qual_query, max_results=5)
    strategy_result = tavily_client.search(strategy_query, max_results=4)
    failure_result = tavily_client.search(failure_query, max_results=5)
    micro_result = tavily_client.search(micro_query, max_results=4)
    industry_result = tavily_client.search(industry_query, max_results=4)
    news_result = tavily_client.search(news_query, max_results=5)

    macro_text = " ".join([macro_result.answer, *macro_result.snippets]).lower()
    macro_score = 32.0
    keyword_impacts = {
        "recession": 16,
        "credit tightening": 11,
        "high interest": 9,
        "rate hike": 9,
        "default": 12,
        "demand slowdown": 8,
        "inflation": 5,
        "uncertainty": 6,
    }
    for phrase, impact in keyword_impacts.items():
        if phrase in macro_text:
            macro_score += impact

    def _clean_snippet(text: str, max_len: int = 320) -> str:
        clipped = " ".join(str(text or "").split())
        return clipped[:max_len].strip()

    qual_snippets = [_clean_snippet(s, 320) for s in qual_result.snippets if str(s).strip()]
    qual_snippets = [s for s in qual_snippets if len(s) >= 40][:8]
    macro_notes = [_clean_snippet(macro_result.answer, 220)] + [_clean_snippet(s, 220) for s in macro_result.snippets[:3]]
    micro_notes = [_clean_snippet(micro_result.answer, 220)] + [_clean_snippet(s, 220) for s in micro_result.snippets[:3]]
    industry_notes = [_clean_snippet(industry_result.answer, 220)] + [_clean_snippet(s, 220) for s in industry_result.snippets[:3]]
    news_notes = [_clean_snippet(news_result.answer, 220)] + [_clean_snippet(s, 220) for s in news_result.snippets[:4]]

    macro_notes = [x for x in macro_notes if x]
    micro_notes = [x for x in micro_notes if x]
    industry_notes = [x for x in industry_notes if x]
    news_notes = [x for x in news_notes if x]

    return {
        "macro_stress_score": max(0.0, min(100.0, macro_score)),
        "macro_notes": macro_notes,
        "micro_notes": micro_notes,
        "industry_notes": industry_notes,
        "news_notes": news_notes,
        "qual_snippets": qual_snippets,
        "sources": list(
            dict.fromkeys(
                macro_result.sources
                + qual_result.sources
                + strategy_result.sources
                + failure_result.sources
                + micro_result.sources
                + industry_result.sources
                + news_result.sources
            )
        ),
        "source_groups": {
            "macro": list(dict.fromkeys(macro_result.sources)),
            "qualitative": list(dict.fromkeys(qual_result.sources)),
            "strategy": list(dict.fromkeys(strategy_result.sources)),
            "failure_check": list(dict.fromkeys(failure_result.sources)),
            "micro": list(dict.fromkeys(micro_result.sources)),
            "industry": list(dict.fromkeys(industry_result.sources)),
            "news": list(dict.fromkeys(news_result.sources)),
        },
        "strategy_notes": [strategy_result.answer] if strategy_result.answer else [],
        "failure_check": {
            "answer": failure_result.answer,
            "snippets": failure_result.snippets,
            "sources": failure_result.sources,
        },
    }


def _collect_peer_metrics(peer_tickers: List[str], macro_stress_score: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ticker in peer_tickers:
        info = fetch_company_info(ticker)
        metrics = compute_metrics(fetch_financials(ticker), company_info=info)
        core_quality = _core_metric_quality(metrics)
        metrics, estimated = _impute_survivor_defaults(metrics, ticker)
        score, _ = MultiFactorRiskEngine(metrics, macro_stress_score).compute_score()
        rows.append(
            {
                "ticker": ticker,
                "metrics": metrics,
                "risk_score": score,
                "estimated": estimated,
                "core_quality": core_quality,
            }
        )
    return rows


def _layer_signals(layers: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    return {
        "macro": list(layers.get("macro", {}).get("signals", [])),
        "business_model": list(layers.get("business_model", {}).get("signals", [])),
        "financial_health": list(layers.get("financial_health", {}).get("signals", [])),
        "operational": list(layers.get("operational", {}).get("signals", [])),
        "qualitative": list(layers.get("qualitative", {}).get("signals", [])),
    }


def _chart_metric_gaps(metric_gaps: Dict[str, object]) -> alt.Chart:
    rows = []
    for k, v in metric_gaps.items():
        raw_gap = float(v or 0.0)
        scaled_gap = 0.0
        if raw_gap != 0:
            sign = 1.0 if raw_gap > 0 else -1.0
            scaled_gap = sign * (abs(raw_gap) ** 0.35)
        rows.append(
            {
                "Metric": _friendly_metric_name(k.replace("_gap", "")),
                "Scaled Gap": scaled_gap,
                "Raw Gap": raw_gap,
            }
        )
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(
                "Metric:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelAngle=-18, labelColor="#94a3b8", labelLimit=180, labelPadding=8),
            ),
            y=alt.Y("Scaled Gap:Q", title="Relative Gap (scaled for readability)"),
            color=alt.condition(alt.datum["Scaled Gap"] >= 0, alt.value("#f97316"), alt.value("#14b8a6")),
            tooltip=["Metric", alt.Tooltip("Raw Gap:Q", format=",.3f"), alt.Tooltip("Scaled Gap:Q", format=",.3f")],
        )
        .properties(height=260)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#94a3b8",
            titleColor="#94a3b8",
            labelFontSize=13,
            titleFontSize=13,
            gridColor="rgba(255,255,255,0.08)",
        )
        .configure_axisX(labelAngle=-18, labelColor="#94a3b8", titleColor="#94a3b8", labelPadding=8)
        .configure_axisY(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure_legend(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure(background="rgba(0,0,0,0)")
    )


def _chart_layer_stress_heatmap(
    layers: Dict[str, Dict[str, object]],
    intelligence: Dict[str, object],
    qual: Dict[str, object],
) -> alt.Chart:
    rows = _layer_stress_rows(layers, intelligence, qual)
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_rect(cornerRadius=8)
        .encode(
            x=alt.X("Layer:N", title=None, sort=["Macro", "Business", "Financial", "Operational", "Qualitative"]),
            color=alt.Color(
                "Stress Score:Q",
                scale=alt.Scale(range=["#d7eef6", "#8fc6d6", "#377ea3", "#ff7eb6"]),
                title="Stress Intensity",
            ),
            tooltip=["Layer", alt.Tooltip("Stress Score:Q", format=".2f"), "Signals"],
        )
        .properties(height=90)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#94a3b8", titleColor="#94a3b8", labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure(background="rgba(0,0,0,0)")
    )


def _layer_stress_rows(
    layers: Dict[str, Dict[str, object]],
    intelligence: Dict[str, object],
    qual: Dict[str, object],
) -> List[Dict[str, object]]:
    def _coerce_score(raw: object) -> Optional[float]:
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if val < 0:
            return 0.0
        # Support both 0-1 and 0-100 style inputs.
        if val > 1.5 and val <= 100:
            return max(0.0, min(1.0, val / 100.0))
        return max(0.0, min(1.0, val))

    layer_titles = {
        "macro": "Macro",
        "business_model": "Business",
        "financial_health": "Financial",
        "operational": "Operational",
        "qualitative": "Qualitative",
    }
    macro_norm = max(0.0, min(1.0, float(intelligence.get("macro_stress_score", 0.0) or 0.0) / 100.0))
    qual_norm = max(0.0, min(1.0, float(qual.get("distress_intensity", 0.0) or 0.0) / 10.0))
    rows: List[Dict[str, object]] = []
    for key, title in layer_titles.items():
        layer_payload = dict(layers.get(key, {}) or {})
        signal_items = [str(x).strip() for x in list(layer_payload.get("signals", []) or []) if str(x).strip()]
        signal_count = len(signal_items)
        explicit = _coerce_score(layer_payload.get("score"))
        # Fallback from signal density when no explicit score exists.
        score = explicit if explicit is not None else min(1.0, signal_count * 0.30)
        if key == "macro":
            score = max(score, macro_norm)
        if key == "qualitative":
            score = max(score, qual_norm)
        rows.append(
            {
                "Layer": title,
                "Stress Score": round(max(0.0, min(1.0, score)), 3),
                "Signals": signal_count,
                "Signal Details": " | ".join(signal_items[:3]) if signal_items else "No strong signal",
            }
        )
    return rows


def _chart_risk_contribution(components: Dict[str, float]) -> go.Figure:
    """Interactive Plotly horizontal bar for risk contribution."""
    import plotly.graph_objects as go
    import plotly.express as px
    if not components:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=180)
        return fig
    items = sorted(components.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [float(v) for _, v in items]
    total = sum(values) or 1
    pcts = [v / total * 100 for v in values]
    colors = ["#ff7eb6" if v > 0.3 else "#f97316" if v > 0.15 else "#0f62fe" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pcts], textposition="outside",
        textfont=dict(color="#f4f4f4", size=11, family="Inter"),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<br>% of total: <br><extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        margin=dict(l=0, r=50, t=30, b=0), height=max(200, len(labels)*40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8"),
        yaxis=dict(showgrid=False, color="#94a3b8"),
        title=dict(text="Risk Contribution Decomposition", font=dict(color="#e0e0e0", size=13)),
    )
    return fig


def _chart_nlp_theme_scores(qual: Dict[str, object]) -> go.Figure:
    """Interactive Plotly NLP theme severity chart."""
    import plotly.graph_objects as go
    import plotly.express as px
    theme_scores = dict(qual.get("theme_scores", {}) or {})
    theme_counts = dict(qual.get("themes", {}) or {})
    if not theme_scores:
        fig = go.Figure()
        fig.add_annotation(text="No NLP theme signals detected", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color="#94a3b8", size=14))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=180)
        return fig
    items = sorted(theme_scores.items(), key=lambda kv: kv[1], reverse=True)
    themes = [_friendly_theme_name(t) for t, _ in items]
    scores = [float(s) for _, s in items]
    mentions = [int(theme_counts.get(t, 0) or 0) for t, _ in items]
    colors = ["#ff7eb6" if s >= 0.45 else "#f97316" if s >= 0.25 else "#0f62fe" for s in scores]
    fig = go.Figure(go.Bar(
        x=themes, y=scores,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.2f}" for s in scores], textposition="outside",
        textfont=dict(color="#f4f4f4", size=12, family="Inter"),
        customdata=mentions,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<br>Mentions: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        margin=dict(l=0, r=0, t=30, b=0), height=280,
        xaxis=dict(showgrid=False, tickangle=-20, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8",
                   title="Severity Score (0-1)", range=[0, min(1.1, max(scores)*1.3) if scores else 1]),
        title=dict(text="NLP Distress Theme Scores", font=dict(color="#e0e0e0", size=14)),
    )
    return fig


def _chart_before_after(original: float, adjusted: float) -> alt.Chart:
    df = pd.DataFrame(
        [
            {"Scenario": "Original", "Risk": float(original)},
            {"Scenario": "Counterfactual", "Risk": float(adjusted)},
        ]
    )
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Scenario:N", title=None, sort=["Original", "Counterfactual"]),
            y=alt.Y("Risk:Q", title="Risk Score"),
            color=alt.Color("Scenario:N", scale=alt.Scale(range=["#ff7eb6", "#14b8a6"]), legend=None),
            tooltip=["Scenario", "Risk"],
        )
        .properties(height=220)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="#94a3b8",
            titleColor="#94a3b8",
            labelFontSize=13,
            titleFontSize=13,
            gridColor="rgba(255,255,255,0.08)",
        )
        .configure_axisX(labelAngle=0, labelColor="#94a3b8", titleColor="#94a3b8")
        .configure_axisY(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure_legend(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure(background="rgba(0,0,0,0)")
    )


def _chart_risk_components(components: Dict[str, float]) -> go.Figure:
    """Interactive Plotly donut for risk component mix."""
    import plotly.graph_objects as go
    import plotly.express as px
    if not components:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=200)
        return fig
    labels = [k.replace("_", " ").title() for k in components]
    values = [abs(float(v)) for v in components.values()]
    colors = ["#0f62fe", "#14b8a6", "#f97316", "#7dd3fc", "#78a9ff", "#ff7eb6", "#3ddbd9"]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors[:len(labels)], line=dict(color="#080d1a", width=2)),
        textinfo="label+percent",
        textfont=dict(color="#e0e0e0", size=11, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Score: %{value:.3f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        margin=dict(l=0, r=0, t=10, b=0), height=260,
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Risk Component Mix", font=dict(color="#e0e0e0", size=13)),
    )
    return fig


def _chart_component_delta(failing_components: Dict[str, float], survivor_components: Dict[str, float]) -> go.Figure:
    """Interactive Plotly grouped bar for component delta vs survivors."""
    import plotly.graph_objects as go
    import plotly.express as px
    all_keys = sorted(set(list(failing_components.keys()) + list(survivor_components.keys())))
    if not all_keys:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=200)
        return fig
    labels = [k.replace("_", " ").title() for k in all_keys]
    fail_vals = [float(failing_components.get(k, 0)) for k in all_keys]
    surv_vals = [float(survivor_components.get(k, 0)) for k in all_keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Failed Company", x=labels, y=fail_vals,
                         marker=dict(color="#ff7eb6", line=dict(width=0)),
                         hovertemplate="<b>%{x}</b><br>Failed: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(name="Survivor Avg", x=labels, y=surv_vals,
                         marker=dict(color="#3ddbd9", line=dict(width=0)),
                         hovertemplate="<b>%{x}</b><br>Survivor: %{y:.3f}<extra></extra>"))
    fig.update_layout(
        barmode="group", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Inter"),
        margin=dict(l=0, r=0, t=30, b=0), height=280,
        xaxis=dict(showgrid=False, tickangle=-18, color="#94a3b8"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8", title="Component Score"),
        legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Component Delta vs Survivor Cohort", font=dict(color="#e0e0e0", size=13)),
    )
    return fig


def _chart_peer_positioning(
    failing_ticker: str,
    failing_metrics: Dict[str, Optional[float]],
    failing_risk_score: float,
    survivor_rows: List[Dict[str, object]],
) -> alt.Chart:
    rows = []
    rows.append(
        {
            "Ticker": failing_ticker,
            "Current Ratio": float(failing_metrics.get("current_ratio") or 0.01),
            "Debt/Equity": float(failing_metrics.get("debt_to_equity") or 0.01),
            "Risk Score": float(failing_risk_score),
            "Group": "Failing",
        }
    )
    for row in survivor_rows:
        m = row.get("metrics", {})
        rows.append(
            {
                "Ticker": str(row.get("ticker")),
                "Current Ratio": float((m or {}).get("current_ratio") or 0.01),
                "Debt/Equity": float((m or {}).get("debt_to_equity") or 0.01),
                "Risk Score": float(row.get("risk_score") or 0.0),
                "Group": "Survivor",
            }
        )

    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_circle(size=170, opacity=0.85)
        .encode(
            x=alt.X("Current Ratio:Q", title="Current Ratio (higher is better)"),
            y=alt.Y("Debt/Equity:Q", title="Debt/Equity (lower is better)"),
            color=alt.Color("Group:N", scale=alt.Scale(range=["#ff7eb6", "#14b8a6"])),
            size=alt.Size("Risk Score:Q", scale=alt.Scale(range=[80, 520])),
            tooltip=["Ticker", "Group", alt.Tooltip("Current Ratio:Q", format=".2f"), alt.Tooltip("Debt/Equity:Q", format=".2f"), "Risk Score"],
        )
        .properties(height=280)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelColor="#94a3b8", titleColor="#94a3b8", labelFontSize=12, titleFontSize=12, gridColor="rgba(255,255,255,0.08)")
        .configure_legend(labelColor="#94a3b8", titleColor="#94a3b8")
        .configure(background="rgba(0,0,0,0)")
    )


def _glossary() -> Dict[str, str]:
    return {
        "Risk Score": "Composite 0-100 probability-like distress score built from debt, liquidity, growth, burn, and macro stress.",
        "Current Ratio": "Current assets divided by current liabilities. Below 1 generally means tighter liquidity.",
        "Debt/Equity": "Leverage ratio. Higher values indicate heavier debt burden relative to equity.",
        "Cash Burn": "Cash consumed by operations. Lower burn is healthier under stress.",
        "Counterfactual": "A simulated alternative world where the failing company adopts survivor-like metrics.",
        "Model Lab": "Local in-app analyst model used as a second opinion to stabilize reasoning output.",
    }


def _render_glossary_panel() -> None:
    glossary = _glossary()
    st.markdown("#### Glossary")
    st.caption("Click a term to see what it means.")

    terms = list(glossary.keys())
    if "glossary_selected" not in st.session_state or st.session_state["glossary_selected"] not in glossary:
        st.session_state["glossary_selected"] = terms[0]

    with st.container(key="glossary_panel"):
        row_size = 3
        for start in range(0, len(terms), row_size):
            row_terms = terms[start:start + row_size]
            cols = st.columns(len(row_terms))
            for col, term in zip(cols, row_terms):
                if col.button(term, key=f"glossary_btn_{term}", use_container_width=True, type="secondary"):
                    st.session_state["glossary_selected"] = term

    selected = str(st.session_state.get("glossary_selected", terms[0]))
    description = glossary.get(selected, "")
    st.markdown(
        f"<div class='glossary-card'><strong>{html.escape(selected)}</strong><br/>{html.escape(description)}</div>",
        unsafe_allow_html=True,
    )


def _render_workflow_trace(
    *,
    profile_name: str,
    ticker: str,
    failed: bool,
    failure_status: Dict[str, object],
    peers: Dict[str, object],
    survivor_tickers: List[str],
    intelligence: Dict[str, object],
    reasoning: Dict[str, object],
) -> None:
    st.markdown("### SignalForge Workflow")
    st.caption("End-to-end execution trace from input to final recommendations.")
    st.markdown(
        """
        <div class="flow-grid">
          <div class="flow-card"><b>1) Input Resolve</b><br/>Company name/ticker is mapped to canonical entity profile.</div>
          <div class="flow-card"><b>2) Evidence Gathering</b><br/>Macro + micro + industry + news + strategy + failure evidence fetched.</div>
          <div class="flow-card"><b>3) Failure Gate</b><br/>LLM classifier + evidence guardrails decide failed/not-failed.</div>
          <div class="flow-card"><b>4) Peer Matching</b><br/>Industry-family/business-model matching ranks comparable survivors.</div>
          <div class="flow-card"><b>5) Risk Engines</b><br/>Layered analysis + multi-factor risk score + local analyst model.</div>
          <div class="flow-card"><b>6) Counterfactual Twin</b><br/>Replace weak metrics with survivor averages and recompute risk.</div>
          <div class="flow-card"><b>7) Reasoning Synthesis</b><br/>LLM + deterministic rules write causes, deltas, actions.</div>
          <div class="flow-card"><b>8) Evidence Report</b><br/>Simple + Analyst + Evidence tabs + export JSON/Markdown.</div>
          <div class="flow-card"><b>9) Ask Me Q&A</b><br/>Question-aware reasoning grounded in current report context.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "```text\n"
        "Input -> Resolve -> Tavily Evidence -> Failure Verification -> Peer/Survivor Match\n"
        "      -> Layered Stress + Risk Score -> Counterfactual Twin -> Strategy Synthesis\n"
        "      -> Analyst Report + Exports -> Ask Me (context + web evidence)\n"
        "```"
    )

    with st.expander("1) Input + Entity Resolution", expanded=False):
        st.write(f"- Input resolved to: **{profile_name} ({ticker})**")
        st.write(f"- Target matching profile: sector `{peers.get('sector', 'Unknown')}`, industry `{peers.get('industry', 'Unknown')}`")

    with st.expander("2) Evidence Gathering (Web Intelligence)", expanded=True):
        st.write("- We gather macro, micro, industry, strategy, and failure-check signals with Tavily.")
        source_groups = dict(intelligence.get("source_groups", {}) or {})
        labels = {
            "macro": "Macro",
            "micro": "Micro (Company)",
            "industry": "Industry",
            "news": "News",
            "failure_check": "Failure Check",
            "strategy": "Strategy",
            "qualitative": "Qualitative",
        }
        for key in ["macro", "micro", "industry", "news", "failure_check", "strategy", "qualitative"]:
            st.write(f"**{labels[key]} Sources**")
            urls = list(source_groups.get(key, []) or [])
            if not urls:
                st.write("- No source captured")
            else:
                for url in urls[:4]:
                    st.write(f"- {url}")

    with st.expander("3) Failure Verification Gate", expanded=False):
        st.write(f"- Classified as failed/distressed: **{failed}**")
        st.write(f"- Confidence: **{float(failure_status.get('confidence', 0.0))*100:.1f}%**")
        st.write(
            f"- Provider / Model: `{failure_status.get('provider_used', 'fallback')}` / "
            f"`{failure_status.get('model_used', 'fallback')}`"
        )
        st.write(f"- Rationale: {str(failure_status.get('reason', ''))}")

    with st.expander("4) Peer + Survivor Benchmarking", expanded=False):
        st.write(f"- Peer model family: `{peers.get('industry_family', 'other')}`")
        st.write(f"- Selected survivor cohort: **{', '.join(survivor_tickers)}**")
        st.write("- Survivors are prioritized by business-model similarity + lower observed risk score.")

    with st.expander("5) Scoring + Counterfactual Twin", expanded=False):
        st.write("- Compute layered stress + composite risk score.")
        st.write("- Replace failing metrics with survivor averages to simulate prevention pathway.")

    with st.expander("6) Strategy Synthesis", expanded=False):
        st.write("- Final recommendations combine deterministic metric gaps + LLM narrative synthesis.")
        for item in list(reasoning.get("prevention_measures", []) or [])[:3]:
            st.write(f"- {item}")


def _humanize_gap_terms(text: str) -> str:
    out = str(text or "")
    replacements = {
        "debt_to_equity_gap": "debt-to-equity gap",
        "current_ratio_gap": "liquidity buffer gap (current ratio)",
        "revenue_growth_gap": "revenue growth gap",
        "cash_burn_gap": "cash burn gap",
    }
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _clean_reasoning_line(text: str, max_len: int = 180) -> str:
    cleaned = str(text or "")
    cleaned = _humanize_gap_terms(cleaned)
    cleaned = cleaned.replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"#+\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -*")
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1].rstrip() + "…"
    return cleaned


def _chat_bubble(text: str, role: str) -> str:
    # Allow typing animation HTML to pass unescaped
    if "<div class='typing-dots'>" in (text or ""):
        safe = text
    else:
        safe = html.escape(text or "").replace("\n", "<br/>")
    if role == "assistant":
        return f"<div class='chat-bubble-ai'>🤖 {safe}</div>"
    return f"<div class='chat-bubble-user'>{safe}</div>"


def _is_typing_message(msg: Dict[str, object]) -> bool:
    if str(msg.get("role", "")) != "assistant":
        return False
    text = str(msg.get("text", "") or "")
    return "<div class='typing-dots'>" in text or text.strip().lower() == "typing..."

def _strengthen_reasoning(
    reasoning: Dict[str, object],
    *,
    deterministic_recommendations: List[str],
    layers: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    out = dict(reasoning)

    failure_drivers = [_clean_reasoning_line(x) for x in list(out.get("failure_drivers", []) or []) if str(x).strip()]
    if len(failure_drivers) < 3:
        layer_backfill: List[str] = []
        for key in ["financial_health", "operational", "business_model", "qualitative", "macro"]:
            layer_backfill.extend(list(layers.get(key, {}).get("signals", [])))
        for signal in layer_backfill:
            if signal not in failure_drivers:
                failure_drivers.append(signal)
            if len(failure_drivers) >= 3:
                break
    out["failure_drivers"] = failure_drivers[:3]

    measures = [_clean_reasoning_line(x) for x in list(out.get("prevention_measures", []) or []) if str(x).strip()]
    for rec in deterministic_recommendations:
        clean_rec = _clean_reasoning_line(rec)
        if clean_rec not in measures:
            measures.append(clean_rec)
        if len(measures) >= 4:
            break
    out["prevention_measures"] = measures[:4]

    survivor_diffs = [_clean_reasoning_line(x) for x in list(out.get("survivor_differences", []) or []) if str(x).strip()]
    out["survivor_differences"] = survivor_diffs[:4]
    technical_notes = [_clean_reasoning_line(x, max_len=220) for x in list(out.get("technical_notes", []) or []) if str(x).strip()]
    out["technical_notes"] = technical_notes[:5]

    if not str(out.get("plain_english_explainer", "")).strip():
        out["plain_english_explainer"] = (
            "This company failed because debt and cash pressure stayed high while revenue momentum weakened. "
            "The benchmark survivors kept stronger liquidity and lower leverage."
        )
    out["plain_english_explainer"] = _clean_reasoning_line(str(out.get("plain_english_explainer", "")), max_len=420)

    if not str(out.get("executive_summary", "")).strip():
        out["executive_summary"] = (
            "Consensus view: distress risk is materially reducible by applying survivor-like balance sheet and liquidity discipline."
        )
    out["executive_summary"] = _clean_reasoning_line(str(out.get("executive_summary", "")), max_len=260)

    return out


def _first_note(notes: List[str], fallback: str) -> str:
    for n in notes:
        text = str(n or "").strip()
        if text:
            return text
    return fallback


def _compose_failure_narrative(
    *,
    company_name: str,
    industry: str,
    reasoning: Dict[str, object],
    layers: Dict[str, Dict[str, object]],
    intelligence: Dict[str, object],
) -> str:
    drivers = [str(x).strip() for x in list(reasoning.get("failure_drivers", []) or []) if str(x).strip()]
    top_driver_text = ", ".join(drivers[:3]) if drivers else "persistent balance-sheet and operating stress"

    macro_signal = _first_note(
        list(intelligence.get("macro_notes", []) or []),
        "Macro stress remained elevated with tighter credit and weaker demand conditions.",
    )
    micro_signal = _first_note(
        list(intelligence.get("micro_notes", []) or []),
        "Company-specific execution and funding risks compounded financial pressure.",
    )
    industry_signal = _first_note(
        list(intelligence.get("industry_notes", []) or []),
        f"{industry} conditions became less forgiving as competition and funding pressure intensified.",
    )

    leadership_keywords = ["management", "leadership", "governance", "execution", "strategy", "oversight"]
    micro_blob = " ".join(list(intelligence.get("micro_notes", []) or []) + list(intelligence.get("news_notes", []) or [])).lower()
    leadership_sentence = (
        "Leadership and execution signals suggest strategic decisions did not correct risk early enough."
        if any(k in micro_blob for k in leadership_keywords)
        else "Execution risk appears to have reinforced the downside once stress started building."
    )

    macro_layer = ", ".join(list(layers.get("macro", {}).get("signals", []))[:2]) or "macro stress stayed elevated"
    financial_layer = ", ".join(list(layers.get("financial_health", {}).get("signals", []))[:2]) or "financial fragility widened"

    return (
        f"{company_name} failed because {top_driver_text}. "
        f"At a system level, {macro_layer}; inside the company, {financial_layer}. "
        f"{leadership_sentence} "
        f"Macro news signal: {macro_signal} "
        f"Micro/company signal: {micro_signal} "
        f"Industry signal: {industry_signal}"
    )


def _compose_prevention_narrative(
    *,
    company_name: str,
    industry: str,
    reasoning: Dict[str, object],
    failing_metrics: Dict[str, Optional[float]],
    comparison: Dict[str, object],
    simulation: Dict[str, object],
    intelligence: Dict[str, object],
) -> str:
    survivor_avg = comparison.get("survivor_average_metrics", {}) or {}
    actions = [str(x).strip() for x in list(reasoning.get("prevention_measures", []) or []) if str(x).strip()]

    cr_now = float(failing_metrics.get("current_ratio") or 0.0)
    cr_target = float(survivor_avg.get("current_ratio") or max(1.2, cr_now))
    de_now = float(failing_metrics.get("debt_to_equity") or 0.0)
    de_target = float(survivor_avg.get("debt_to_equity") or max(1.2, de_now * 0.6 if de_now > 0 else 1.2))
    burn_now = abs(float(failing_metrics.get("cash_burn") or 0.0))
    burn_target = abs(float(survivor_avg.get("cash_burn") or burn_now * 0.45))
    rev_now = float(failing_metrics.get("revenue_growth") or 0.0)
    rev_target = float(survivor_avg.get("revenue_growth") or max(0.03, rev_now))
    before = float(simulation.get("original_score", simulation.get("baseline_score", 0.0)) or 0.0)
    after = float(simulation.get("adjusted_score", before) or before)
    improvement = float(simulation.get("improvement_percentage", 0.0) or 0.0)

    macro_signal = _first_note(
        list(intelligence.get("macro_notes", []) or []),
        "Macro conditions stayed tight, so liquidity and refinancing flexibility mattered most.",
    )
    industry_signal = _first_note(
        list(intelligence.get("industry_notes", []) or []),
        f"{industry} competition and pricing pressure required earlier cost discipline.",
    )

    a1 = actions[0] if len(actions) > 0 else f"build liquidity buffer from {cr_now:.2f} toward {cr_target:.2f}"
    a2 = actions[1] if len(actions) > 1 else f"reduce leverage from debt/equity {de_now:.2f} toward {de_target:.2f}"
    a3 = actions[2] if len(actions) > 2 else "stabilize revenue mix while reducing fixed-cost intensity"

    paragraph = (
        f"{company_name} could likely have avoided collapse with an earlier three-step prevention sequence. "
        f"First, {a1}, because short-term liquidity is the fastest control lever when markets tighten. "
        f"Second, {a2} and lower structural burn (about {burn_now:,.0f} toward {burn_target:,.0f}) to reduce refinancing risk and preserve runway. "
        f"Third, {a3}, targeting revenue growth recovery from {rev_now*100:.1f}% toward {rev_target*100:.1f}% so operating cash flow can normalize. "
        f"The counterfactual simulation supports this path: modeled risk drops from {before:.1f} to {after:.1f} ({improvement:.1f}% improvement). "
        f"Macro signal: {macro_signal} Industry signal: {industry_signal}"
    )
    paragraph = _humanize_gap_terms(paragraph)
    paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
    return paragraph


def _build_analyst_deep_dive(
    *,
    profile_name: str,
    profile_ticker: str,
    profile_industry: str,
    failing_metrics: Dict[str, Optional[float]],
    comparison: Dict[str, object],
    failing_risk_score: float,
    simulation: Dict[str, object],
    layers: Dict[str, Dict[str, object]],
    reasoning: Dict[str, object],
    intelligence: Dict[str, object],
    qual: Dict[str, object],
) -> str:
    survivor_avg = comparison.get("survivor_average_metrics", {}) or {}

    dte = float(failing_metrics.get("debt_to_equity") or 0.0)
    dte_surv = float(survivor_avg.get("debt_to_equity") or 0.0)
    cr = float(failing_metrics.get("current_ratio") or 0.0)
    cr_surv = float(survivor_avg.get("current_ratio") or 0.0)
    rg = float(failing_metrics.get("revenue_growth") or 0.0)
    rg_surv = float(survivor_avg.get("revenue_growth") or 0.0)

    macro_signal = _first_note(
        list(intelligence.get("macro_notes", []) or []),
        "Macro conditions were adverse and credit remained tight.",
    )
    micro_signal = _first_note(
        list(intelligence.get("micro_notes", []) or []),
        "Company-specific execution and funding pressure accelerated the downside.",
    )
    industry_signal = _first_note(
        list(intelligence.get("industry_notes", []) or []),
        f"The {profile_industry} environment showed elevated structural pressure.",
    )
    news_signal = _first_note(
        list(intelligence.get("news_notes", []) or []),
        "News flow pointed to escalating distress signals before collapse.",
    )

    fin_signals = ", ".join(list(layers.get("financial_health", {}).get("signals", []))[:3]) or "balance-sheet stress"
    biz_signals = ", ".join(list(layers.get("business_model", {}).get("signals", []))[:2]) or "demand-side fragility"
    op_signals = ", ".join(list(layers.get("operational", {}).get("signals", []))[:2]) or "operating resilience limits"
    qual_summary = str(qual.get("forensic_summary", "Qualitative pressure remained elevated."))

    prevention = list(reasoning.get("prevention_measures", []) or [])
    p1 = prevention[0] if len(prevention) > 0 else "reduce leverage toward survivor norms"
    p2 = prevention[1] if len(prevention) > 1 else "rebuild liquidity buffers before refinancing windows close"
    p3 = prevention[2] if len(prevention) > 2 else "stabilize demand and cost structure early"

    adjusted = float(simulation.get("adjusted_score", failing_risk_score))
    improvement = float(simulation.get("improvement_percentage", 0.0))

    essay = (
        f"{profile_name} ({profile_ticker}) presents a classic distress progression in which fragile financial structure met an "
        f"unforgiving operating and market environment. The model-estimated risk score of {failing_risk_score:.1f}/100 is not driven "
        f"by one isolated factor; it is the combined result of leverage, liquidity pressure, weakening growth quality, and external stress. "
        f"Relative to survivor peers, leverage appears substantially heavier (debt-to-equity {dte:.2f} vs {dte_surv:.2f}), while near-term "
        f"liquidity is materially thinner (current ratio {cr:.2f} vs {cr_surv:.2f}). Revenue momentum also trails survivor cohorts "
        f"({rg:.2f} vs {rg_surv:.2f}), which reduces flexibility exactly when the balance sheet needs it most.\n\n"
        f"From a layered diagnostics perspective, the financial layer is the strongest contributor to downside ({fin_signals}), while the "
        f"business layer indicates demand fragility ({biz_signals}). Operationally, the system sees {op_signals}. This pattern means the company "
        f"was not only exposed to macro pressure, but also lacked the internal cushion to absorb volatility. The qualitative forensics module "
        f"corroborates this direction: {qual_summary}\n\n"
        f"Macro and external intelligence strengthen the same conclusion. Macro signal: {macro_signal} Micro/company signal: {micro_signal} "
        f"Industry signal: {industry_signal} News sequence signal: {news_signal} Taken together, these indicate that leadership and financing "
        f"decisions likely reacted too late to deteriorating conditions. In practice, once refinancing pressure and confidence erosion begin "
        f"to interact, optionality collapses quickly.\n\n"
        f"The counterfactual twin shows this was not inevitable: if core metrics were aligned to survivor benchmarks, modeled risk falls to "
        f"{adjusted:.1f}/100, an improvement of {improvement:.1f}%. That delta is large enough to imply that prevention required earlier balance-sheet "
        f"discipline and tighter operating controls, not just a better market cycle. The highest-impact path is to {p1}; then {p2}; and finally {p3}. "
        f"This sequence addresses solvency risk first, execution risk second, and long-term resilience third."
    )
    essay = _humanize_gap_terms(essay)
    essay = re.sub(r"[ \t]+", " ", essay)
    essay = re.sub(r"\n{3,}", "\n\n", essay).strip()
    return essay


def _build_report_bundle(
    profile_name: str,
    ticker: str,
    failed: bool,
    reasoning: Dict[str, object],
    failing_risk_score: float,
    simulation: Dict[str, object],
    survivor_tickers: List[str],
    metric_gaps: Dict[str, object],
    qual_summary: Optional[Dict[str, Any]] = None,
    failure_narrative: str = "",
    analyst_deep_dive: str = "",
    prevention_narrative: str = "",
) -> Tuple[str, str]:
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "company": {"name": profile_name, "ticker": ticker, "failed": failed},
        "summary": reasoning,
        "failure_narrative": failure_narrative,
        "prevention_narrative": prevention_narrative,
        "analyst_deep_dive": analyst_deep_dive,
        "scores": {
            "failing_risk_score": failing_risk_score,
            "adjusted_risk_score": simulation.get("adjusted_score"),
            "improvement_percentage": simulation.get("improvement_percentage"),
        },
        "survivor_cohort": survivor_tickers,
        "metric_gaps": metric_gaps,
        "qualitative_forensics": {
            "distress_intensity": (qual_summary or {}).get("distress_intensity"),
            "confidence": (qual_summary or {}).get("confidence"),
            "forensic_summary": (qual_summary or {}).get("forensic_summary"),
            "theme_scores": (qual_summary or {}).get("theme_scores", {}),
            "theme_mentions": (qual_summary or {}).get("themes", {}),
            "keywords": (qual_summary or {}).get("keywords", [])[:15],
        },
    }
    json_text = json.dumps(payload, indent=2)

    md = [
        f"# SignalForge Report: {profile_name} ({ticker})",
        "",
        f"- Failed case verified: **{failed}**",
        f"- Risk score: **{failing_risk_score:.2f}/100**",
        f"- Counterfactual adjusted risk: **{float(simulation.get('adjusted_score', 0)):.2f}/100**",
        f"- Improvement: **{float(simulation.get('improvement_percentage', 0)):.2f}%**",
        "",
        "## Plain-English Summary",
        str(reasoning.get("plain_english_explainer", "")),
        "",
        "## Why It Failed",
    ]
    for item in reasoning.get("failure_drivers", [])[:3]:
        md.append(f"- {item}")
    if failure_narrative:
        md.extend(["", failure_narrative])
    if analyst_deep_dive:
        md.extend(["", "## Analyst Deep Dive", analyst_deep_dive])
    md.extend(["", "## What Survivors Did Differently"])
    for item in reasoning.get("survivor_differences", [])[:3]:
        md.append(f"- {item}")
    md.extend(["", "## Prevention Moves"])
    for item in reasoning.get("prevention_measures", [])[:3]:
        md.append(f"- {item}")
    if prevention_narrative:
        md.extend(["", prevention_narrative])
    md.extend(
        [
            "",
            "## NLP Forensics",
            f"- Distress intensity: **{float((qual_summary or {}).get('distress_intensity', 0.0)):.2f}/10**",
            f"- NLP confidence: **{float((qual_summary or {}).get('confidence', 0.0))*100:.1f}%**",
            f"- Summary: {(qual_summary or {}).get('forensic_summary', 'No qualitative summary available.')}",
        ]
    )
    markdown_text = "\n".join(md)

    return json_text, markdown_text


def _qa_context_from_report(
    *,
    profile_name: str,
    ticker: str,
    industry: str,
    reasoning: Dict[str, object],
    failing_risk_score: float,
    macro_stress_score: float,
    comparison: Dict[str, object],
    simulation: Dict[str, object],
    failing_metrics: Dict[str, Optional[float]],
    survivor_tickers: List[str],
    layers: Dict[str, Dict[str, object]],
    local_before_prob: float,
    local_after_prob: float,
    qual_summary: Optional[Dict[str, Any]] = None,
    intelligence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    key_metrics = ["debt_to_equity", "current_ratio", "cash_burn", "revenue_growth", "revenue"]
    fail_core = {k: failing_metrics.get(k) for k in key_metrics}
    surv_avg = comparison.get("survivor_average_metrics", {}) or {}
    surv_core = {k: surv_avg.get(k) for k in key_metrics}

    return {
        "company": {"name": profile_name, "ticker": ticker, "industry": industry},
        "simple_view": {
            "plain_english_explainer": reasoning.get("plain_english_explainer"),
            "failure_drivers": reasoning.get("failure_drivers", []),
            "survivor_differences": reasoning.get("survivor_differences", []),
            "prevention_measures": reasoning.get("prevention_measures", []),
            "risk_scores": {
                "failing_risk": failing_risk_score,
                "macro_stress": macro_stress_score,
                "adjusted_risk": simulation.get("adjusted_score"),
                "improvement_pct": simulation.get("improvement_percentage"),
            },
        },
        "analyst_view": {
            "survivor_cohort": survivor_tickers,
            "failing_core_metrics": fail_core,
            "survivor_core_metrics": surv_core,
            "metric_gaps": comparison.get("metric_gaps", {}),
            "component_scores": comparison.get("failing_components", {}),
            "layer_signals": {
                "macro": list(layers.get("macro", {}).get("signals", [])),
                "business_model": list(layers.get("business_model", {}).get("signals", [])),
                "financial_health": list(layers.get("financial_health", {}).get("signals", [])),
                "operational": list(layers.get("operational", {}).get("signals", [])),
                "qualitative": list(layers.get("qualitative", {}).get("signals", [])),
            },
            "local_model_probabilities": {
                "before": local_before_prob,
                "after": local_after_prob,
            },
            "qualitative_forensics": {
                "distress_intensity": (qual_summary or {}).get("distress_intensity"),
                "confidence": (qual_summary or {}).get("confidence"),
                "forensic_summary": (qual_summary or {}).get("forensic_summary"),
                "theme_scores": (qual_summary or {}).get("theme_scores", {}),
                "theme_mentions": (qual_summary or {}).get("themes", {}),
                "top_keywords": (qual_summary or {}).get("keywords", [])[:10],
            },
            "macro_micro_industry_news_signals": {
                "macro": list((intelligence or {}).get("macro_notes", []) or [])[:5],
                "micro": list((intelligence or {}).get("micro_notes", []) or [])[:5],
                "industry": list((intelligence or {}).get("industry_notes", []) or [])[:5],
                "news": list((intelligence or {}).get("news_notes", []) or [])[:5],
            },
        },
    }


@st.cache_resource
def _local_model() -> LocalAnalystModel:
    model = LocalAnalystModel(random_state=42)
    model.train(n_samples=7000)
    return model


def main() -> None:
    st.set_page_config(page_title="SignalForge", page_icon="📉", layout="wide")
    _inject_styles()
    _render_header()

    if "analysis_active" not in st.session_state:
        st.session_state["analysis_active"] = False
    if "assistant_open" not in st.session_state:
        st.session_state["assistant_open"] = False
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "text": "I will be your personal AI for this SignalForge Failure Intelligence report.",
            }
        ]
    if "assistant_pending_question" not in st.session_state:
        st.session_state["assistant_pending_question"] = None
    if "assistant_waiting" not in st.session_state:
        st.session_state["assistant_waiting"] = False
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = None
    if "llm_test_result" not in st.session_state:
        st.session_state["llm_test_result"] = None

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    groq_key = os.getenv("GROQ_API_KEY", "")
    watsonx_api_key = os.getenv("WATSONX_API_KEY", "")
    watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "")
    watsonx_url = os.getenv("WATSONX_URL", "")
    watsonx_model = os.getenv("WATSONX_MODEL", "")

    with st.sidebar:
        st.markdown("### Settings")
        reasoning_mode = st.selectbox("Reasoning Mode", ["Collaborative Council (recommended)", "Single Provider"], index=0)
        provider_choice = st.selectbox("LLM Provider", ["IBM watsonx.ai", "Groq"], index=0)

    missing_watsonx = [
        name
        for name, value in [
            ("WATSONX_API_KEY", watsonx_api_key),
            ("WATSONX_PROJECT_ID", watsonx_project_id),
            ("WATSONX_URL", watsonx_url),
            ("WATSONX_MODEL", watsonx_model),
        ]
        if not str(value).strip()
    ]
    groq_client = GroqReasoningClient(groq_key)
    watsonx_client = None if missing_watsonx else WatsonxReasoningClient(
        api_key=watsonx_api_key,
        project_id=watsonx_project_id,
        base_url=watsonx_url,
        model=watsonx_model,
    )

    single_provider_chain = _enabled_provider_chain(
        provider_choice=provider_choice,
        groq_client=groq_client,
        watsonx_client=watsonx_client,
    )
    if not single_provider_chain:
        st.error("No LLM provider is configured. Add Groq or watsonx credentials in `.env`.")
        return

    if provider_choice == "IBM watsonx.ai" and missing_watsonx:
        st.warning(
            "IBM watsonx.ai is missing required configuration: "
            + ", ".join(missing_watsonx)
            + ". Single-provider requests will fail over automatically."
        )
    if provider_choice == "Groq" and not groq_client.enabled:
        st.warning("Groq key is missing. Single-provider requests will fail over to watsonx.ai when available.")

    primary_provider_name, primary_provider_client = single_provider_chain[0]
    reasoning_client = primary_provider_client
    active_provider_name = primary_provider_name
    if primary_provider_name == "IBM watsonx.ai":
        active_model_name = watsonx_model or "unknown"
    else:
        active_model_name = groq_client.models[0] if groq_client.models else "unknown"

    if reasoning_mode == "Collaborative Council (recommended)":
        if watsonx_client is None:
            st.warning("Council mode will run with Groq + Local only because watsonx.ai is not fully configured.")
        active_provider_name = "Collaborative Council"
        active_model_name = (
            f"Groq + watsonx + Local (synth: {watsonx_model})"
            if watsonx_client is not None
            else f"Groq + Local (synth: {groq_client.models[0] if groq_client.models else 'unknown'})"
        )

    with st.sidebar:
        if st.button("Test LLM Connection", use_container_width=True):
            try:
                test_client = reasoning_client if active_provider_name != "Collaborative Council" else (watsonx_client or groq_client)
                test_provider = "IBM watsonx.ai" if test_client is watsonx_client and watsonx_client is not None else "Groq"
                ok, message = _test_llm_connection(test_client, test_provider)
                st.session_state["llm_test_result"] = {"ok": ok, "message": message}
            except Exception as exc:
                normalized = _normalize_provider_error(test_provider, exc)
                if test_provider == "IBM watsonx.ai" and _is_watson_quota_error(exc):
                    normalized = (
                        f"{normalized} This run will fall back to Groq+Local until Watson quota is restored."
                    )
                st.session_state["llm_test_result"] = {"ok": False, "message": normalized}

        llm_test_result = st.session_state.get("llm_test_result")
        if isinstance(llm_test_result, dict):
            if llm_test_result.get("ok"):
                st.success(str(llm_test_result.get("message", "Connection OK")))
            else:
                st.error(str(llm_test_result.get("message", "Connection failed")))

    _render_llm_badge(active_provider_name, active_model_name)

    with st.container(border=True):
        st.markdown("### Start Analysis")
        st.write("Enter a company full name or ticker symbol.")
        company_input = st.text_input("Company Name or Ticker", value="", placeholder="Example: Lehman Brothers or LEHMQ")

        with st.expander("⚙ Advanced Controls", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                max_auto_peers = st.slider("Peer candidate count", 4, 10, 6)
            with col_b:
                survivor_count = st.slider("Survivor benchmark size", 2, 5, 3)
            st.caption("Keys are loaded from `.env` automatically.")

        run_clicked = st.button("Run Failure Forensics", type="primary", use_container_width=True)

    if run_clicked:
        st.session_state["analysis_active"] = True
        st.session_state["analysis_cache"] = None
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "text": "I will be your personal AI for this SignalForge Failure Intelligence report.",
            }
        ]
        st.session_state["assistant_pending_question"] = None
        st.session_state["assistant_waiting"] = False

    if not st.session_state.get("analysis_active", False):
        return

    if not company_input.strip():
        st.error("Please enter a company name or ticker.")
        st.session_state["analysis_active"] = False
        return

    tavily = TavilyClient(tavily_key)
    local_model = _local_model()
    cache_key = {
        "company_input": company_input.strip().upper(),
        "max_auto_peers": max_auto_peers,
        "survivor_count": survivor_count,
        "mode": reasoning_mode,
        "provider": active_provider_name,
        "model": active_model_name,
    }
    cached = st.session_state.get("analysis_cache")
    use_cache = bool(cached) and cached.get("cache_key") == cache_key and not run_clicked

    if use_cache:
        bundle = cached["bundle"]
        profile = bundle["profile"]
        peers = bundle.get("peers", {"sector": profile.sector, "industry": profile.industry, "peers": []})
        intelligence = bundle["intelligence"]
        failure_status = bundle["failure_status"]
        failed = bundle["failed"]
        failing_metrics = bundle["failing_metrics"]
        used_failed_imputation = bundle["used_failed_imputation"]
        macro_stress_score = bundle["macro_stress_score"]
        layers = bundle["layers"]
        failing_risk_score = bundle["failing_risk_score"]
        failing_components = bundle["failing_components"]
        survivor_rows = bundle["survivor_rows"]
        survivor_tickers = bundle["survivor_tickers"]
        comparison = bundle["comparison"]
        simulation = bundle["simulation"]
        recommendations = bundle["recommendations"]
        reasoning = bundle["reasoning"]
        council_output = bundle.get("council_output")
        qual = bundle.get("qual") or qualitative_summary("", intelligence.get("qual_snippets", []))
        local_before = bundle["local_before"]
        local_after = bundle["local_after"]
    else:
        with st.spinner("Resolving company and verifying failure status..."):
            resolved = resolve_company_input(company_input)
            if resolved is None:
                st.error("Could not resolve that company input.")
                st.session_state["analysis_active"] = False
                return

            profile = fetch_company_profile(resolved.ticker)
            info = fetch_company_info(profile.ticker)
            intelligence = _fetch_tavily_intelligence(tavily, profile.name, profile.ticker, profile.industry)

            failure_status, verify_errors, _ = _invoke_with_provider_failover(
                provider_chain=single_provider_chain,
                method_name="verify_failure_status",
                payload={
                    "company_input": company_input,
                    "resolved_name": profile.name,
                    "ticker": profile.ticker,
                    "tavily_answer": str(intelligence["failure_check"]["answer"]),
                    "tavily_snippets": list(intelligence["failure_check"]["snippets"]),
                },
                validator=lambda row: "is_failed" in row,
            )
            if failure_status is None:
                failure_status = _fallback_failure_status(verify_errors)

        failed = bool(failure_status.get("is_failed", False))
        # NOTE: we do NOT block the analysis when is_failed=False.
        # Companies like "Apple in 1990" or high-risk companies that haven't
        # formally failed yet still need forensic analysis. We show a warning
        # banner but always continue to the full report.

        progress = st.progress(0, text="Collecting financial and peer data...")
        failing_metrics = compute_metrics(fetch_financials(profile.ticker), company_info=info)
        if _core_metric_quality(failing_metrics) < 2:
            failing_metrics, used_failed_imputation = _impute_failed_defaults(failing_metrics)
        else:
            used_failed_imputation = False

        qual = qualitative_summary("", intelligence["qual_snippets"], metrics=failing_metrics)
        raw_qual_intensity = float(qual.get("distress_intensity", sum(qual["themes"].values())))
        qualitative_intensity = max(0.0, min(6.0, raw_qual_intensity * 0.62))
        macro_stress_score = float(intelligence["macro_stress_score"])

        layer_engine = LayeredAnalysisEngine(failing_metrics, qual["themes"], macro_stress_score)
        layers = layer_engine.analyze_all_layers()
        failing_risk_score, failing_components = MultiFactorRiskEngine(failing_metrics, macro_stress_score).compute_score()

        progress.progress(40, text="Building survivor benchmark...")
        peers = find_peer_companies(profile.ticker, max_peers=max_auto_peers)
        peer_rows = _collect_peer_metrics([p["ticker"] for p in peers["peers"]], macro_stress_score)
        if not peer_rows:
            st.error("Unable to collect peer data.")
            return

        peer_meta = {str(p.get("ticker")): p for p in peers.get("peers", [])}
        for row in peer_rows:
            meta = peer_meta.get(str(row.get("ticker")), {})
            row["match_type"] = str(meta.get("match_type", "fit"))
            row["match_reason"] = str(meta.get("match_reason", "Peer fit"))
            row["match_source"] = str(meta.get("match_source", "unknown"))
            try:
                row["match_score"] = float(meta.get("match_score", 0))
            except (TypeError, ValueError):
                row["match_score"] = 0.0

        high_fit_pool = [x for x in peer_rows if float(x.get("match_score", 0.0)) >= 60.0]
        selection_pool = high_fit_pool if high_fit_pool else peer_rows
        healthy_pool = [x for x in selection_pool if not _looks_distressed_symbol(str(x.get("ticker", "")))]
        if healthy_pool:
            selection_pool = healthy_pool

        def _survivor_sort_key(row: Dict[str, object]) -> Tuple[float, float, bool, int]:
            return (
                -float(row.get("match_score", 0.0)),
                float(row.get("risk_score", 100.0)),
                bool(row.get("estimated", False)),
                -int(row.get("core_quality", 0)),
            )

        better = [x for x in selection_pool if x["risk_score"] < failing_risk_score]
        survivor_rows = (
            sorted(better, key=_survivor_sort_key)[:survivor_count]
            if better
            else sorted(selection_pool, key=_survivor_sort_key)[:survivor_count]
        )

        survivor_tickers = [x["ticker"] for x in survivor_rows]
        survivor_metrics = [x["metrics"] for x in survivor_rows]

        comparison = compare_failure_vs_survivors(failing_metrics, survivor_metrics, macro_stress_score)
        simulation = simulate_counterfactual(failing_metrics, comparison["survivor_average_metrics"], macro_stress_score)
        recommendations = generate_strategy_recommendations(failing_metrics, comparison["survivor_average_metrics"])

        local_before = local_model.predict(failing_metrics, macro_stress_score, qualitative_intensity)
        local_after = local_model.predict(simulation["adjusted_metrics"], macro_stress_score, qualitative_intensity)

        progress.progress(74, text="Running LLM and local analyst reasoning...")
        council_output = None
        if reasoning_mode == "Collaborative Council (recommended)":
            evidence_bundle = _build_council_evidence_bundle(intelligence)
            council_output = run_reasoning_council(
                {
                    "company_profile": {
                        "name": profile.name,
                        "ticker": profile.ticker,
                        "industry": profile.industry,
                        "sector": profile.sector,
                    },
                    "metrics": failing_metrics,
                    "peer_summary": {
                        **comparison,
                        "survivor_tickers": survivor_tickers,
                        "layer_signals": {
                            "macro": list(layers.get("macro", {}).get("signals", [])),
                            "business_model": list(layers.get("business_model", {}).get("signals", [])),
                            "financial_health": list(layers.get("financial_health", {}).get("signals", [])),
                            "operational": list(layers.get("operational", {}).get("signals", [])),
                            "qualitative": list(layers.get("qualitative", {}).get("signals", [])),
                        },
                    },
                    "evidence_bundle": evidence_bundle,
                    "simulation": simulation,
                    "recommendations": recommendations,
                    "failing_risk_score": failing_risk_score,
                    "macro_stress_score": macro_stress_score,
                    "qualitative_intensity": qualitative_intensity,
                    "failure_year": None,
                    "groq_client": groq_client,
                    "watsonx_client": watsonx_client,
                    "local_model": local_model,
                    "synthesis_provider": "watsonx" if watsonx_client is not None else "groq",
                }
            )
            reasoning = _legacy_reasoning_from_council(council_output)
        else:
            reasoning, reasoning_errors, _ = _invoke_with_provider_failover(
                provider_chain=single_provider_chain,
                method_name="generate_reasoning",
                payload={
                    "company_name": profile.name,
                    "ticker": profile.ticker,
                    "industry": profile.industry,
                    "failing_risk_score": failing_risk_score,
                    "survivor_tickers": survivor_tickers,
                    "layer_signals": {
                        "macro": list(layers.get("macro", {}).get("signals", [])),
                        "business_model": list(layers.get("business_model", {}).get("signals", [])),
                        "financial_health": list(layers.get("financial_health", {}).get("signals", [])),
                        "operational": list(layers.get("operational", {}).get("signals", [])),
                        "qualitative": list(layers.get("qualitative", {}).get("signals", [])),
                    },
                    "metric_gaps": comparison["metric_gaps"],
                    "simulation": simulation,
                    "recommendations": recommendations,
                    "tavily_notes": (
                        intelligence["macro_notes"]
                        + intelligence["micro_notes"]
                        + intelligence["industry_notes"]
                        + intelligence["news_notes"]
                        + intelligence["strategy_notes"]
                    ),
                },
                validator=lambda row: bool(
                    str(row.get("plain_english_explainer", "")).strip()
                    or str(row.get("executive_summary", "")).strip()
                ),
            )
            if reasoning is None:
                reasoning = _fallback_reasoning(recommendations, reasoning_errors)

        reasoning = _strengthen_reasoning(
            reasoning,
            deterministic_recommendations=recommendations,
            layers=layers,
        )

        technical_notes = list(reasoning.get("technical_notes", []) or [])
        technical_notes.insert(
            0,
            (
                f"Local model distress probability changes from {local_before.risk_probability*100:.1f}% "
                f"to {local_after.risk_probability*100:.1f}% under the counterfactual."
            ),
        )
        reasoning["technical_notes"] = [_clean_reasoning_line(x, max_len=220) for x in technical_notes[:4]]
        progress.progress(100, text="Report ready.")

        st.session_state["analysis_cache"] = {
            "cache_key": cache_key,
            "bundle": {
                "profile": profile,
                "peers": peers,
                "intelligence": intelligence,
                "failure_status": failure_status,
                "failed": failed,
                "failing_metrics": failing_metrics,
                "used_failed_imputation": used_failed_imputation,
                "macro_stress_score": macro_stress_score,
                "layers": layers,
                "failing_risk_score": failing_risk_score,
                "failing_components": failing_components,
                "survivor_rows": survivor_rows,
                "survivor_tickers": survivor_tickers,
                "comparison": comparison,
                "simulation": simulation,
                "recommendations": recommendations,
                "reasoning": reasoning,
                "council_output": council_output,
                "qual": qual,
                "local_before": local_before,
                "local_after": local_after,
            },
        }

    st.markdown("### Verification")
    v1, v2, v3 = st.columns([1.6, 1, 1])
    v1.write(f"Resolved: **{profile.name} ({profile.ticker})**")
    v2.metric("Failure Confidence", f"{float(failure_status.get('confidence', 0.0))*100:.1f}%")
    v3.write(
        f"Provider: `{failure_status.get('provider_used', 'fallback')}` | "
        f"Model: `{failure_status.get('model_used', 'fallback')}`"
    )

    failed = bool(failure_status.get("is_failed", False))
    if failed:
        st.markdown('<span class="badge badge-failed">Failed/Distressed Case Confirmed</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-ok">Likely Not Failed</span>', unsafe_allow_html=True)

    st.write(str(failure_status.get("reason", "")))
    for e in failure_status.get("evidence", [])[:3]:
        st.write(f"- {e}")

    if not failed:
        st.warning("This case does not appear to be failed/distressed. Enter a failed company to generate a full forensic report.")
        st.session_state["analysis_active"] = False
        return

    if reasoning_mode == "Collaborative Council (recommended)" and council_output:
        wx_error = str((((council_output.get("model_breakdown", {}) or {}).get("watsonx", {}) or {}).get("errors") or "").strip())
        if wx_error:
            if _is_watson_quota_error(wx_error):
                st.info(
                    "IBM watsonx.ai is configured, but quota/permission is currently blocked. "
                    "This run automatically used Groq + Local with the same Tavily signal bundle."
                )
            else:
                st.warning(
                    "IBM watsonx.ai was temporarily unavailable in this run. "
                    "The council automatically continued with Groq + Local."
                )

    failure_narrative = _compose_failure_narrative(
        company_name=profile.name,
        industry=profile.industry,
        reasoning=reasoning,
        layers=layers,
        intelligence=intelligence,
    )
    prevention_narrative = _compose_prevention_narrative(
        company_name=profile.name,
        industry=profile.industry,
        reasoning=reasoning,
        failing_metrics=failing_metrics,
        comparison=comparison,
        simulation=simulation,
        intelligence=intelligence,
    )
    analyst_deep_dive = _build_analyst_deep_dive(
        profile_name=profile.name,
        profile_ticker=profile.ticker,
        profile_industry=profile.industry,
        failing_metrics=failing_metrics,
        comparison=comparison,
        failing_risk_score=failing_risk_score,
        simulation=simulation,
        layers=layers,
        reasoning=reasoning,
        intelligence=intelligence,
        qual=qual,
    )

    report_json, report_md = _build_report_bundle(
        profile.name,
        profile.ticker,
        failed,
        reasoning,
        failing_risk_score,
        simulation,
        survivor_tickers,
        comparison["metric_gaps"],
        qual,
        failure_narrative,
        analyst_deep_dive,
        prevention_narrative,
    )

    st.markdown("---")
    st.markdown("## Report")

    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "Export JSON Report",
            data=report_json,
            file_name=f"signalforge_{profile.ticker.lower()}_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "Export Markdown Report",
            data=report_md,
            file_name=f"signalforge_{profile.ticker.lower()}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tabs = st.tabs(["Overview", "Analyst View", "Council Output", "Council Trace", "Scenario Lab", "Evidence"])

    with tabs[0]:
        # ── Overview: headline metrics + plain-English story
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Failure Risk", f"{failing_risk_score:.2f}/100",
                  delta=f"vs survivor {float(comparison.get('survivor_average_risk_score', 0)):.1f}")
        v2.metric("Altman Z-Score", f"{float(failing_metrics.get('altman_z', 0.0)):.2f}",
                  delta="< 1.81 = distress" if float(failing_metrics.get("altman_z", 2.0)) < 1.81 else "> 2.99 = safe")
        v3.metric("NLP Distress", f"{float(qual.get('distress_intensity',0)):.1f}/10")
        v4.metric("Council Confidence",
                  f"{float(council_output.get('overall_confidence', 0.0))*100:.0f}%" if council_output else "N/A")

        fail_text = str(reasoning.get("plain_english_explainer", "") or "").strip()
        if fail_text:
            st.markdown(f"<div class='explain' style='margin-top:0.8rem;'>{html.escape(fail_text)}</div>",
                        unsafe_allow_html=True)

        # ── Why it failed (driver bullets)
        drivers = list((council_output or {}).get("failure_drivers", []) or reasoning.get("why_it_failed", []) or [])
        if drivers:
            items = ""
            for item in drivers[:6]:
                d = _coerce_claim_text(item, ["driver", "strategy", "action"])
                conf_value = _coerce_claim_confidence(item)
                conf_html = f" — <span style=color:#ff7eb6>{conf_value*100:.0f}% confidence</span>" if conf_value is not None else ""
                items += f"<div class='council-driver'>{html.escape(d)}{conf_html}</div>"
            st.markdown(
                f"<div class='council-card' style='margin-top:0.8rem;'>"
                f"<div class='council-model-badge'>⚡ Why It Failed</div>{items}</div>",
                unsafe_allow_html=True)

        # ── What would have prevented it
        strategies = list((council_output or {}).get("survivor_strategies", []) or reasoning.get("how_it_could_have_been_prevented", []) or [])
        if strategies:
            items = ""
            for item in strategies[:5]:
                s = _coerce_claim_text(item, ["strategy", "action", "driver"])
                items += f"<div class='council-strategy'>{html.escape(s)}</div>"
            st.markdown(
                f"<div class='council-card' style='border-color:rgba(61,219,217,0.3);'>"
                f"<div class='council-model-badge' style='background:rgba(61,219,217,0.15);color:#6ee7b7;border-color:rgba(61,219,217,0.3);'>✓ How It Could Have Been Prevented</div>{items}</div>",
                unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Failure Risk", f"{failing_risk_score:.2f}/100")
        s2.metric("Adjusted Risk", f"{float(simulation['adjusted_score']):.2f}/100")
        s3.metric("Risk Improvement", f"{float(simulation['improvement_percentage']):.2f}%")
        s4.metric("Local Analyst", f"{local_before.risk_probability*100:.1f}%")
        st.caption(
            f"NLP distress intensity: {float(qual.get('distress_intensity', 0.0)):.1f}/10 "
            f"(confidence {float(qual.get('confidence', 0.0))*100:.0f}%)."
        )
        st.caption(
            "Counterfactual data is simulated (not historical): it replaces failed-company metrics with matched survivor averages."
        )

        c_left, c_right = st.columns(2)
        with c_left:
            st.altair_chart(_chart_before_after(failing_risk_score, float(simulation["adjusted_score"])), use_container_width=True)
        with c_right:
            st.altair_chart(_chart_metric_gaps(comparison["metric_gaps"]), use_container_width=True)

        _render_glossary_panel()

        st.markdown("#### Why It Failed")
        for item in reasoning.get("failure_drivers", [])[:3]:
            st.write(f"- {item}")
        st.markdown(f"<div class='explain'>{failure_narrative}</div>", unsafe_allow_html=True)

        st.markdown("#### Signal Board (Macro, Micro, Industry, News)")
        sig_cols = st.columns(4)
        signal_groups = [
            ("Macro", intelligence.get("macro_notes", [])),
            ("Micro", intelligence.get("micro_notes", [])),
            ("Industry", intelligence.get("industry_notes", [])),
            ("News", intelligence.get("news_notes", [])),
        ]
        for col, (title, notes) in zip(sig_cols, signal_groups):
            col.markdown(f"**{title}**")
            for line in _format_signal_items(list(notes or []), max_items=2):
                col.write(f"- {line}")

        st.markdown("#### How It Could Have Been Prevented")
        for item in reasoning.get("prevention_measures", [])[:3]:
            st.write(f"- {item}")
        st.markdown(f"<div class='explain'>{html.escape(prevention_narrative)}</div>", unsafe_allow_html=True)

    with tabs[1]:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Risk Score", f"{failing_risk_score:.2f}")
        a2.metric("Survivor Avg", f"{float(comparison['survivor_score']):.2f}")
        a3.metric("Macro Stress", f"{macro_stress_score:.1f}")
        a4.metric("Local Post-Fix", f"{local_after.risk_probability*100:.1f}%")

        peer_meta = {str(p.get("ticker")): p for p in peers.get("peers", [])}
        cohort_text = []
        for row in survivor_rows:
            ticker = str(row.get("ticker", ""))
            meta = peer_meta.get(ticker, {})
            match_type = str(meta.get("match_type", row.get("match_type", "fit"))).replace("_", " ")
            cohort_text.append(f"{ticker} ({match_type})")
        st.markdown(f"**Survivor Cohort:** {', '.join(cohort_text) if cohort_text else ', '.join(survivor_tickers)}")
        st.caption(
            f"Peer matching target: sector `{peers.get('sector', 'Unknown')}`, "
            f"industry `{peers.get('industry', 'Unknown')}`, "
            f"family `{peers.get('industry_family', 'other')}`."
        )
        with st.expander("Why these survivors were selected", expanded=False):
            for row in survivor_rows:
                ticker = str(row.get("ticker", ""))
                meta = peer_meta.get(ticker, {})
                reason = str(meta.get("match_reason", row.get("match_reason", "Peer fit")))
                st.write(f"- {ticker}: {reason}")

        st.markdown("#### Analyst Deep Dive")
        essay_html = html.escape(analyst_deep_dive).replace("\n\n", "<br/><br/>")
        st.markdown(f"<div class='explain'>{essay_html}</div>", unsafe_allow_html=True)

        st.markdown("#### Core Metrics")
        t1, t2 = st.columns(2)
        fail_table, survivor_table = _aligned_metric_tables(
            _clean_metrics(failing_metrics),
            _clean_metrics(comparison["survivor_average_metrics"]),
        )
        with t1:
            st.write("Failing Company")
            if fail_table.empty:
                st.warning("No concrete failing-company metrics available for this symbol.")
            else:
                st.dataframe(fail_table, use_container_width=True)
        with t2:
            st.write("Survivor Average")
            if survivor_table.empty:
                st.warning("No survivor metrics available.")
            else:
                st.dataframe(survivor_table, use_container_width=True)

        st.markdown("#### Risk Component Mix")
        st.plotly_chart(_chart_risk_components(failing_components), use_container_width=True, config={"displayModeBar": "hover"})

        st.markdown("#### Component Delta vs Survivors")
        st.plotly_chart(_chart_component_delta(comparison.get("failing_components",{}), comparison.get("survivor_components",{})), use_container_width=True, config={"displayModeBar":"hover"})

        st.markdown("#### Risk Contribution Decomposition")
        st.plotly_chart(_chart_risk_contribution(failing_components), use_container_width=True, config={"displayModeBar":"hover"})

        st.markdown("#### Layer Stress Heatmap")
        # Plotly heatmap for layer stress
        import plotly.graph_objects as go
        import plotly.express as px
        _layer_rows = _layer_stress_rows(layers, intelligence, qual)
        _layer_names = [str(r.get("Layer", "")) for r in _layer_rows]
        _layer_scores = [float(r.get("Stress Score", 0.0) or 0.0) for r in _layer_rows]
        _layer_signals = [str(r.get("Signal Details", "No strong signal")) for r in _layer_rows]
        _layer_counts = [int(r.get("Signals", 0) or 0) for r in _layer_rows]
        _hm_fig = go.Figure(go.Bar(
            x=_layer_names, y=_layer_scores,
            marker=dict(
                color=_layer_scores,
                colorscale=[[0, "#3ddbd9"], [0.4, "#f97316"], [1, "#ff7eb6"]],
                line=dict(width=0), showscale=True,
                colorbar=dict(title="Stress", tickfont=dict(color="#94a3b8"), titlefont=dict(color="#94a3b8")),
            ),
            text=[f"{s:.2f}" for s in _layer_scores], textposition="outside",
            textfont=dict(color="#f4f4f4", size=13, family="Inter"),
            customdata=list(zip(_layer_counts, _layer_signals)),
            hovertemplate="<b>%{x}</b><br>Stress Score: %{y:.2f}<br>Signal count: %{customdata[0]}<br>Signals: %{customdata[1]}<extra></extra>",
        ))
        _hm_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=0, r=0, t=30, b=0), height=260,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", title="Stress Score", range=[0, 1.15]),
            title=dict(text="Layer Stress Heatmap (hover for signal details)", font=dict(color="#e0e0e0", size=13)),
        )
        st.plotly_chart(_hm_fig, use_container_width=True, config={"displayModeBar": "hover"})

        # Plotly interactive peer positioning bubble chart
        _peer_fig_rows = []
        for _pr in survivor_rows[:8]:
            _peer_fig_rows.append({
                "Company": str(_pr.get("ticker", "")),
                "Risk": float(_pr.get("risk_score", 0)),
                "DE": float((_pr.get("metrics") or {}).get("debt_to_equity", 1.5) or 1.5),
                "CR": float((_pr.get("metrics") or {}).get("current_ratio", 1.2) or 1.2),
                "Type": "Survivor",
            })
        _peer_fig_rows.append({
            "Company": profile.ticker,
            "Risk": failing_risk_score,
            "DE": float(failing_metrics.get("debt_to_equity", 3.0) or 3.0),
            "CR": float(failing_metrics.get("current_ratio", 0.8) or 0.8),
            "Type": "Subject",
        })
        _pf_df = pd.DataFrame(_peer_fig_rows)
        _pf_fig = px.scatter(
            _pf_df, x="DE", y="CR", size="Risk", color="Type", text="Company",
            color_discrete_map={"Survivor": "#3ddbd9", "Subject": "#ff7eb6"},
            size_max=40, title="Peer Positioning: Debt/Equity vs Liquidity",
            labels={"DE": "Debt / Equity", "CR": "Current Ratio (Liquidity)", "Risk": "Risk Score"},
        )
        _pf_fig.update_traces(textposition="top center", textfont=dict(color="#f4f4f4", size=10))
        _pf_fig.update_layout(
            plot_bgcolor="rgba(255,255,255,0.02)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=0, r=0, t=40, b=0), height=320,
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8",
                       title="Debt / Equity (lower = safer)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8",
                       title="Current Ratio (higher = safer)"),
            legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        )
        _pf_fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(190,149,255,0.5)",
                           annotation_text="CR=1.0 danger zone", annotation_font_color="#fcd34d")
        _pf_fig.add_vline(x=3.0, line_dash="dash", line_color="rgba(255,126,182,0.4)",
                           annotation_text="DTE=3.0 high risk", annotation_font_color="#fca5a5")
        st.plotly_chart(_pf_fig, use_container_width=True, config={"displayModeBar": "hover"})

        st.markdown("#### NLP Distress Forensics")
        q1, q2, q3 = st.columns(3)
        q1.metric("Distress Intensity", f"{float(qual.get('distress_intensity', 0.0)):.2f}/10")
        q2.metric("NLP Confidence", f"{float(qual.get('confidence', 0.0))*100:.1f}%")
        q3.metric("Positive vs Negated", f"{int(qual.get('positive_mentions', 0))} / {int(qual.get('negated_total', 0))}")

        st.plotly_chart(_chart_nlp_theme_scores(qual), use_container_width=True, config={"displayModeBar": "hover"})
        kdf = pd.DataFrame({"Top Keywords": list(qual.get("keywords", []))[:15]})
        if not kdf.empty:
            st.dataframe(kdf, use_container_width=True)
        st.write(str(qual.get("forensic_summary", "")))

        st.markdown("#### Layered Signals")
        total_layer_signals = sum(len(list((layers.get(k, {}) or {}).get("signals", []) or [])) for k in ["macro", "business_model", "financial_health", "operational", "qualitative"])
        st.caption(
            f"{total_layer_signals} explicit layer signals were used in the model pipeline "
            "(risk scoring + layered diagnostics + council reasoning)."
        )
        lcols = st.columns(5)
        names = ["macro", "business_model", "financial_health", "operational", "qualitative"]
        titles = ["Macro", "Business", "Financial", "Operational", "Qualitative"]
        stress_rows = _layer_stress_rows(layers, intelligence, qual)
        stress_lookup = {str(r.get("Layer", "")): float(r.get("Stress Score", 0.0) or 0.0) for r in stress_rows}
        for i, key in enumerate(names):
            lcols[i].markdown(f"**{titles[i]}**")
            lcols[i].caption(f"Stress: {stress_lookup.get(titles[i], 0.0):.2f}")
            channels, evidence_lines, used_text = _layer_context_details(
                layer_key=key,
                intelligence=intelligence,
                qual=qual,
                failing_metrics=failing_metrics,
            )
            lcols[i].write(f"- Used in: {used_text}")
            lcols[i].write(f"- Source channels: {', '.join(channels)}")
            signals = list(layers.get(key, {}).get("signals", []))
            if not signals:
                lcols[i].write("- No strong signal")
            else:
                for s in signals[:3]:
                    lcols[i].write(f"- {s}")
            for ev in evidence_lines[:2]:
                lcols[i].write(f"- {ev}")

        if used_failed_imputation:
            st.info("Some failed-company metrics were estimated due limited filing availability for this symbol.")

    with tabs[2]:
        if reasoning_mode == "Collaborative Council (recommended)" and council_output:
            _render_council_tab(council_output)
        else:
            st.info("Council mode is off. Switch Reasoning Mode to `Collaborative Council (recommended)` to view collaborative output.")

    with tabs[3]:
        if reasoning_mode == "Collaborative Council (recommended)" and council_output:
            _render_council_trace_tab(council_output, qual=qual)
        else:
            st.info("Council mode is off. Switch Reasoning Mode to `Collaborative Council (recommended)` to inspect the multi-system trace.")

    with tabs[4]:
        st.markdown("### Interactive Scenario Lab")
        st.caption("Adjust strategic levers and instantly see simulated risk impact. Hover each control for meaning.")

        s_col1, s_col2 = st.columns(2)
        with s_col1:
            dte = st.slider(
                "Debt / Equity Ratio",
                min_value=0.2, max_value=6.0,
                value=float(failing_metrics.get("debt_to_equity") or 3.1),
                step=0.1, key="scen_dte",
                help="🎯 Altman Z-Score: DTE > 3.0 = HIGH distress. Target: < 1.5 (survivor avg).",
            )
            cr = st.slider(
                "Current Ratio (Liquidity)",
                min_value=0.3, max_value=3.0,
                value=float(failing_metrics.get("current_ratio") or 0.8),
                step=0.05, key="scen_cr",
                help="🎯 CR < 1.0 = liquidity crisis. Target: > 1.5 to match survivor cohort.",
            )
        with s_col2:
            burn_m = st.slider(
                "Annual Cash Burn ($M)",
                min_value=0, max_value=500,
                value=int((failing_metrics.get("cash_burn") or 250_000_000.0) / 1_000_000),
                step=10, key="scen_burn",
                help="🎯 Burn > 20% of revenue is unsustainable. Reduce toward break-even.",
            )
            rev_growth = st.slider(
                "Revenue Growth Rate",
                min_value=-0.5, max_value=0.4,
                value=float(failing_metrics.get("revenue_growth") or -0.1),
                step=0.02, key="scen_rev",
                help="🎯 Negative growth compounds distress. Survivors averaged +8% growth.",
            )

        custom_metrics = dict(failing_metrics)
        custom_metrics["debt_to_equity"] = dte
        custom_metrics["current_ratio"] = cr
        custom_metrics["cash_burn"] = float(burn_m) * 1_000_000.0
        custom_metrics["revenue_growth"] = rev_growth
        if custom_metrics.get("revenue") is None:
            custom_metrics["revenue"] = 1_000_000_000.0

        custom_score, _ = MultiFactorRiskEngine(custom_metrics, macro_stress_score).compute_score()
        custom_improvement = ((failing_risk_score - custom_score) / max(failing_risk_score, 1e-6)) * 100

        c_metric1, c_metric2, c_metric3 = st.columns(3)
        c_metric1.metric("Original Risk", f"{failing_risk_score:.2f}")
        c_metric2.metric("Scenario Risk", f"{custom_score:.2f}")
        c_metric3.metric("Scenario Improvement", f"{custom_improvement:.2f}%")
        # Plotly interactive scenario comparison
        import plotly.graph_objects as go
        fig_scenario = go.Figure()
        colors = ["#ff7eb6" if failing_risk_score > 50 else "#f97316", "#3ddbd9"]
        labels = ["Current Risk", "Scenario Risk"]
        values = [failing_risk_score, custom_score]
        for i, (lbl, val, col) in enumerate(zip(labels, values, colors)):
            fig_scenario.add_trace(go.Bar(
                x=[lbl], y=[val], name=lbl,
                marker=dict(color=col, line=dict(width=0)),
                text=[f"{val:.1f}"], textposition="outside",
                textfont=dict(color="#f4f4f4", size=16, family="Inter"),
                hovertemplate=f"<b>{lbl}</b><br>Risk Score: %{{y:.2f}}<extra></extra>",
            ))
        fig_scenario.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Inter"),
            margin=dict(l=0, r=0, t=20, b=0), height=280,
            xaxis=dict(showgrid=False, color="#94a3b8"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)", color="#94a3b8",
                       range=[0, max(values)*1.25], title="Risk Score"),
            showlegend=False, barmode="group",
        )
        st.plotly_chart(fig_scenario, use_container_width=True, config={"displayModeBar": False})

        # ── What-if narrative ──────────────────────────────────────────────
        improvement = custom_improvement
        if improvement > 0:
            if dte < float(failing_metrics.get("debt_to_equity") or 3.1):
                st.markdown(
                    f"<div class='council-strategy'>Reducing Debt/Equity from "
                    f"{float(failing_metrics.get('debt_to_equity') or 3.1):.1f}x → {dte:.1f}x cuts "
                    f"leverage risk, freeing cash for operations and reducing covenant breach probability.</div>",
                    unsafe_allow_html=True)
            if cr > float(failing_metrics.get("current_ratio") or 0.8):
                st.markdown(
                    f"<div class='council-strategy'>Improving Liquidity (CR {float(failing_metrics.get('current_ratio') or 0.8):.2f} → {cr:.2f}) "
                    f"reduces short-term default risk and buys runway for restructuring.</div>",
                    unsafe_allow_html=True)
            if rev_growth > float(failing_metrics.get("revenue_growth") or -0.1):
                st.markdown(
                    f"<div class='council-strategy'>Revenue growth improvement ({float(failing_metrics.get('revenue_growth') or -0.1)*100:.0f}% → {rev_growth*100:.0f}%) "
                    f"signals demand recovery — the strongest predictor of avoiding bankruptcy.</div>",
                    unsafe_allow_html=True)
        else:
            st.markdown("<div class='council-driver'>These scenario parameters increase risk relative to the baseline. Try lowering debt or improving liquidity.</div>",
                unsafe_allow_html=True)

    with tabs[5]:
        _render_workflow_trace(
            profile_name=profile.name,
            ticker=profile.ticker,
            failed=failed,
            failure_status=failure_status,
            peers=peers,
            survivor_tickers=survivor_tickers,
            intelligence=intelligence,
            reasoning=reasoning,
        )

        st.markdown("### Failure Verification Evidence")
        st.write(str(intelligence["failure_check"]["answer"]))
        for line in intelligence["failure_check"]["snippets"][:6]:
            st.write(f"- {line}")

        st.markdown("### Macro / Micro / Industry / News Signals")
        st.write("**Macro**")
        for line in _format_signal_items(list(intelligence.get("macro_notes", []) or []), max_items=3):
            st.write(f"- {line}")
        st.write("**Micro (Company-Specific)**")
        for line in _format_signal_items(list(intelligence.get("micro_notes", []) or []), max_items=3):
            st.write(f"- {line}")
        st.write("**Industry Knowledge**")
        for line in _format_signal_items(list(intelligence.get("industry_notes", []) or []), max_items=3):
            st.write(f"- {line}")
        st.write("**News Timeline Signals**")
        for line in _format_signal_items(list(intelligence.get("news_notes", []) or []), max_items=3):
            st.write(f"- {line}")

        st.markdown("### NLP Evidence Digest")
        st.write(str(qual.get("forensic_summary", "")))

        theme_evidence = dict(qual.get("theme_evidence", {}) or {})
        shown_theme = False
        for theme, snippets in theme_evidence.items():
            if not snippets:
                continue
            shown_theme = True
            st.markdown(f"**{_friendly_theme_name(theme)}**")
            for item in snippets[:2]:
                st.write(f"- {item}")
        if not shown_theme:
            st.write("No high-confidence qualitative evidence snippets found.")

        st.markdown("### Tavily Strategy Notes")
        if intelligence["strategy_notes"]:
            for note in intelligence["strategy_notes"]:
                st.write(f"- {note}")
        else:
            st.write("No strategy notes returned.")

        st.markdown("### Source Trace")
        if intelligence["sources"]:
            for url in intelligence["sources"]:
                st.write(f"- {url}")

    qa_context = _qa_context_from_report(
        profile_name=profile.name,
        ticker=profile.ticker,
        industry=profile.industry,
        reasoning=reasoning,
        failing_risk_score=failing_risk_score,
        macro_stress_score=macro_stress_score,
        comparison=comparison,
        simulation=simulation,
        failing_metrics=failing_metrics,
        survivor_tickers=survivor_tickers,
        layers=layers,
        local_before_prob=local_before.risk_probability,
        local_after_prob=local_after.risk_probability,
        qual_summary=qual,
        intelligence=intelligence,
    )

    if not st.session_state.get("assistant_open", False):
        with st.container(key="jarvis_trigger"):
            if st.button("💬 Ask Me", key="jarvis_open_btn", use_container_width=True):
                st.session_state["assistant_open"] = True
                st.rerun()
    else:
        _inject_assistant_panel_mode_style(False)
        with st.container(key="jarvis_panel"):
            hdr_left, hdr_right = st.columns([5.2, 1.2])
            hdr_left.markdown("<div class='jarvis-title'>SignalForge AI</div>", unsafe_allow_html=True)
            hdr_left.markdown("<div class='jarvis-sub'>Personal AI for this report</div>", unsafe_allow_html=True)
            if hdr_right.button("✕", key="jarvis_close_btn", type="secondary", use_container_width=True):
                st.session_state["assistant_open"] = False
                st.rerun()

            for msg in st.session_state.get("assistant_messages", [])[-8:]:
                text_content = str(msg.get("text", ""))
                role = str(msg.get("role", "assistant"))
                if "<div class='typing-dots'>" in text_content:
                    html_str = f"<div class='chat-bubble-ai'>🤖 SignalForge AI is thinking... <div class='typing-dots'><span></span><span></span><span></span></div></div>"
                    st.html(html_str)
                else:
                    st.markdown(_chat_bubble(text_content, role), unsafe_allow_html=True)

            # Use st.chat_input for immediate clearing on submit
            if ask_q := st.chat_input("Ask about this report...", key="jarvis_chat_input"):
                if not st.session_state.get("assistant_waiting", False):
                    q = ask_q.strip()
                    st.session_state["assistant_messages"].append({"role": "user", "text": q})
                    st.session_state["assistant_messages"].append({"role": "assistant", "text": "<div class='typing-dots'><span></span><span></span><span></span></div>"})
                    st.session_state["assistant_pending_question"] = q
                    st.session_state["assistant_waiting"] = True
                    st.rerun()

            pending_q = st.session_state.get("assistant_pending_question")
            if st.session_state.get("assistant_waiting", False) and pending_q:
                # Calculate silently, letting the typing bubble show instead of a spinner
                if True:
                    try:
                        web_search_1 = tavily.search(
                            f"{profile.name} {profile.ticker} {pending_q}",
                            max_results=4,
                        )
                        web_search_2 = tavily.search(
                            f"{profile.industry} distressed company survivor strategies {pending_q}",
                            max_results=4,
                        )
                        web_evidence = []
                        for snippet, source in zip(web_search_1.snippets[:4], web_search_1.sources[:4]):
                            web_evidence.append({"snippet": snippet, "source": source})
                        for snippet, source in zip(web_search_2.snippets[:4], web_search_2.sources[:4]):
                            web_evidence.append({"snippet": snippet, "source": source})

                        # Build a rich knowledge-base system context for the chatbot
                        failure_knowledge_base = """
You are SignalForge AI — a specialist in corporate failure forensics, financial distress, and turnaround strategy.
You have been trained on:

## Bankruptcy Prediction Models
- **Altman Z-Score**: Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5. Z < 1.81 distress zone, 1.81–2.99 grey, > 2.99 safe.
- **Ohlson O-Score**: Logistic model using 9 factors including firm size, leverage, liquidity, and performance.
- **Zmijewski Model**: Focuses on ROA, leverage, and liquidity as the three strongest failure predictors.
- **Campbell-Hilscher-Szilagyi (2008)**: Market-based distress probability from equity volatility + book leverage.

## The 5 Failure Archetypes
1. **Liquidity Squeeze**: Current ratio < 1.0, negative working capital, revolving credit exhausted → Lehman, SVB, Bear Stearns.
2. **Debt Death Spiral**: Debt/Equity > 3x, covenant breaches, refinancing wall → Enron, TXU Energy Future, iHeartMedia.
3. **Revenue Collapse**: Revenue declining > 15% YoY, negative operating leverage → Blockbuster, Kodak, RadioShack.
4. **Fraud / Governance Failure**: Accounting manipulation, SEC investigation, sudden auditor departure → WorldCom, Theranos, FTX.
5. **Disruption Obsolescence**: Business model disruption + failure to pivot + cash burn → Borders, Sears, Toys R Us.

## Key Metric Thresholds for Distress
- Debt/Equity > 3.0: HIGH risk
- Current Ratio < 0.8: CRITICAL liquidity
- Cash Burn > 20% revenue: UNSUSTAINABLE
- Revenue Growth < -10%: SEVERE demand decline
- Operating Margin < -5%: STRUCTURAL loss
- Interest Coverage < 1.5x: DEFAULT risk

## Survivor Best Practices
- Survivors rebalanced to Debt/Equity < 1.5 within 18 months of stress signal
- Maintained > 6 months cash runway at all times
- Diversified revenue streams to reduce single-product/channel concentration
- Reduced fixed costs by 15-25% while preserving R&D investment
- Secured revolving credit facilities BEFORE they were needed

## Analysis Framework
When answering, always:
1. Reference the specific metric values from the report context
2. Map findings to one of the 5 archetypes
3. Cite the risk score and what drives it
4. Give a precise, actionable recommendation with timeline
5. Acknowledge uncertainty where present
"""
                        answer, answer_errors, provider_used = _invoke_with_provider_failover(
                            provider_chain=single_provider_chain,
                            method_name="answer_report_question",
                            payload={
                                "question": pending_q,
                                "report_context": qa_context,
                                "web_evidence": web_evidence,
                                "system_knowledge": failure_knowledge_base,
                            },
                            validator=lambda row: bool(str(row.get("answer", "")).strip()),
                        )
                        if answer is None:
                            answer = _fallback_answer(answer_errors)
                            provider_used = "fallback"
                        answer_text = str(answer.get("answer", "")).strip() or "I recommend starting with immediate liquidity stabilization."
                        rationale = str(answer.get("rationale", "")).strip()
                        if rationale:
                            answer_text = f"{answer_text}\n\nWhy: {rationale}"
                        if provider_used:
                            answer_text = f"{answer_text}\n\nSource model: {provider_used}"
                    except Exception:
                        answer_text = "I could not complete that request right now. Please try again."

                msgs = list(st.session_state.get("assistant_messages", []) or [])
                typing_idx: Optional[int] = None
                for idx in range(len(msgs) - 1, -1, -1):
                    if _is_typing_message(dict(msgs[idx] or {})):
                        typing_idx = idx
                        break
                if typing_idx is not None:
                    msgs[typing_idx] = {"role": "assistant", "text": answer_text}
                    msgs = [m for j, m in enumerate(msgs) if j == typing_idx or not _is_typing_message(dict(m or {}))]
                else:
                    msgs.append({"role": "assistant", "text": answer_text})
                st.session_state["assistant_messages"] = msgs
                st.session_state["assistant_pending_question"] = None
                st.session_state["assistant_waiting"] = False
                st.rerun()

    if reasoning_mode == "Collaborative Council (recommended)":
        caption_model = active_model_name
        workflow_line = "Workflow: Groq draft -> watsonx critique/synthesis -> Local NLP/Analyst sanity check"
        if council_output:
            wx_error = str((((council_output.get("model_breakdown", {}) or {}).get("watsonx", {}) or {}).get("errors") or "").strip())
            if wx_error:
                workflow_line = "Workflow: Groq draft -> Local sanity + Groq synthesis fallback (watsonx unavailable this run)"
        st.caption(
            f"Reasoning mode: {reasoning_mode} | Systems: {caption_model} | "
            f"{workflow_line} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    else:
        st.caption(
            f"Reasoning mode: {reasoning_mode} | Provider: {reasoning.get('provider_used', active_provider_name)} | "
            f"Model: {reasoning.get('model_used', 'fallback')} | "
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    main()
