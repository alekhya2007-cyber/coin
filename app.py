# app.py — Coinie 🪙 Personal Finance Chatbot (single-file website)
# Run: python -m streamlit run app.py
# Deps: streamlit, requests  (optional: transformers for local embeddings)
# If you have a Hugging Face token: set it in the sidebar or hard-code below.

import os
import json
import math
import time
from datetime import datetime, date
import requests
import streamlit as st

# ==============================
# Optional: local embeddings (your snippet)
# ==============================
EMBED_TOKENIZER = None
EMBED_MODEL = None
EMBED_OK = False
try:
    from transformers import AutoTokenizer, AutoModel
    try:
        EMBED_TOKENIZER = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-english-r2")
        EMBED_MODEL = AutoModel.from_pretrained("ibm-granite/granite-embedding-english-r2")
        EMBED_OK = True
    except Exception as _e_load:
        EMBED_OK = False
except Exception:
    EMBED_OK = False

def embed_texts(texts):
    """Return list of embeddings (simple mean pooling). Falls back to zeros if embeddings unavailable."""
    if not EMBED_OK:
        return [[0.0]*128 for _ in texts]
    import torch
    vecs = []
    for t in texts:
        inputs = EMBED_TOKENIZER(t, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = EMBED_MODEL(**inputs)
            # Mean-pool last_hidden_state
            emb = out.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()
            vecs.append(emb)
    return vecs

def cosine(a, b):
    import math
    da = sum(x*x for x in a) ** 0.5
    db = sum(y*y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da*db)

# ==============================
# UI CONFIG & STYLE
# ==============================
st.set_page_config(page_title="Coinie 🪙", page_icon="🪙", layout="wide")

PRIMARY_YELLOW = "#FFD84D"
st.markdown(f"""
<style>
:root {{
  --accent: {PRIMARY_YELLOW};
}}
.block-container {{ padding-top: 1.4rem !important; }}
.co-card {{
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  background: rgba(255,255,255,0.55);
  border: 1px solid rgba(255,255,255,0.4);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}}
.co-badge {{
  display:inline-block;padding:.2rem .6rem;border-radius:999px;
  background:rgba(255,216,77,.2);border:1px dashed var(--accent);font-size:.85rem;
}}
.co-underline {{
  background: linear-gradient(120deg, rgba(255,216,77,.35) 0%, rgba(255,255,255,0) 85%);
  padding:0 .25rem;border-radius:6px;
}}
div.stButton>button {{ border-radius:999px; }}
</style>
""", unsafe_allow_html=True)

# ==============================
# STATE
# ==============================
if "smart_pot" not in st.session_state:
    st.session_state.smart_pot = {
        "mode": "With Transaction",  # or "Without Transaction"
        "round_multiple": 10,
        "goal_amount": 5000.0,
        "savings_total": 0.0,
        "history": []  # {ts, kind, amount, multiple, contribution, note}
    }

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# ==============================
# MODEL HELPERS (HF Inference API)
# ==============================
def hf_generate(prompt, model_id, hf_token, max_new_tokens=220, temperature=0.4, top_p=0.9, retries=3):
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_full_text": False
        }
    }

    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=300)  # 5 min timeout
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                return str(data)
            else:
                err = f"⚠️ Model error {r.status_code}: {r.text}"
        except requests.exceptions.RequestException as e:
            err = f"⚠️ Request failed: {e}"

        # Retry unless last attempt
        if attempt < retries - 1:
            time.sleep(10)
            continue
        return err


def simple_sentiment_hf(text, hf_token):
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    try:
        r = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)
        if r.status_code != 200:
            return {"error": f"{r.status_code}: {r.text}"}
        out = r.json()
        if isinstance(out, list) and out and isinstance(out[0], list):
            pred = max(out[0], key=lambda x: x.get("score", 0))
            return {"label": pred.get("label"), "score": round(pred.get("score", 0), 3)}
        elif isinstance(out, list) and out and "label" in out[0]:
            return {"label": out[0]["label"], "score": round(out[0]["score"], 3)}
        return {"raw": out}
    except Exception as e:
        return {"error": str(e)}

def naive_keywords(text, top_k=10):
    stop = {"about","with","that","this","there","their","would","could","should","where","while","which","being","from","have","into","your","they","them","then","than","after","before","because","over","under","between","during","these","those","here","every"}
    words = [w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()]
    bag = {}
    for w in words:
        if len(w) >= 5 and w not in stop and w.isalpha():
            bag[w] = bag.get(w, 0) + 1
    return [k for k,_ in sorted(bag.items(), key=lambda x: (-x[1], x[0]))[:top_k]]

# ==============================
# PROMPTS (Granite-flavored)
# ==============================
SYSTEM = (
    "You are Coinie 🪙, a friendly educational personal finance assistant. "
    "Give clear, practical, **non-advisory** guidance (not legal/financial advice). "
    "Tone adapts: playful/simple for students; concise/practical for professionals. "
    "Always end with a one-line disclaimer."
)

def prompt_qa(user_type, question):
    tone = "Playful, simple language with small relatable examples (e.g., Maggi packets)." if user_type=="Student" else \
           "Professional, concise, practical guidance referencing SIP/retirement where helpful."
    return (
        f"{SYSTEM}\nUser type: {user_type}\nTone: {tone}\n\n"
        f"Question:\n{question}\n\n"
        "Answer with: a short summary, 3–5 bullet steps, and 1-line disclaimer."
    )

def prompt_budget(user_type, income, categories):
    return (
        f"{SYSTEM}\nCreate a budget summary for a {user_type}.\n"
        f"Monthly income: {income}\nExpenses JSON: {json.dumps(categories, ensure_ascii=False)}\n\n"
        "Output: surplus/deficit, savings rate, suggested 50/30/20 baseline (tweak if needed), "
        "3–5 actionable optimizations, and a 1-line disclaimer."
    )

def prompt_insights(user_type, categories):
    return (
        f"{SYSTEM}\nAnalyze spending for a {user_type}.\n"
        f"Categories (₹/month): {json.dumps(categories, ensure_ascii=False)}\n\n"
        "Output: Top 3 categories to optimize (with % impact), quick wins & medium-term steps, "
        "KPI checklist, and a 1-line disclaimer."
    )

# ==============================
# SMART POT
# ==============================
WITTY = [
    "Every coin counts — even your snacks fund future-you. 🥤➡️🪙",
    "Small rounds, big dreams. Keep rolling! 🔄",
    "Coinie stamped another micro-win. 📈",
    "Tiny habits compound into wow. ✨",
    "Round here, round there — hello, savings! 💛",
]

def next_round_contribution(amount, multiple):
    """If already on a multiple, go to the *next* multiple (never zero)."""
    if multiple <= 0:
        return 0.0
    remainder = amount % multiple
    if remainder == 0:
        return float(multiple)
    return round(multiple - remainder, 2)

def add_pot_entry(kind, amount, multiple, note=""):
    contrib = next_round_contribution(amount, multiple)
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": kind, "amount": round(float(amount),2),
        "multiple": multiple, "contribution": contrib, "note": note
    }
    st.session_state.smart_pot["history"].append(entry)
    st.session_state.smart_pot["savings_total"] = round(
        st.session_state.smart_pot["savings_total"] + contrib, 2
    )
    return entry

def add_daily_without_txn(multiple):
    """Without Transaction mode: add fixed multiple even with no spend."""
    contrib = float(multiple)
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "kind": "daily",
        "amount": 0.0,
        "multiple": multiple,
        "contribution": contrib,
        "note": "Daily auto-save (Without Transaction mode)"
    }
    st.session_state.smart_pot["history"].append(entry)
    st.session_state.smart_pot["savings_total"] = round(
        st.session_state.smart_pot["savings_total"] + contrib, 2
    )
    return entry

def pot_progress():
    total = st.session_state.smart_pot["savings_total"]
    goal = st.session_state.smart_pot["goal_amount"]
    pct = 0 if goal<=0 else min(100, round((total/goal)*100,1))
    return total, goal, pct

# ==============================
# SIDEBAR (settings)
# ==============================
with st.sidebar:
    st.markdown("## 🪙 Coinie Settings")
    st.markdown('<div class="co-card">', unsafe_allow_html=True)

    # Hugging Face
    hf_token = st.text_input("Hugging Face Token (READ)", os.getenv("HF_API_KEY",""), type="password",
                             help="Create at https://huggingface.co/settings/tokens")
    model_id = st.text_input(
        "Model (Inference API)",
        value="mistralai/Mistral-7B-Instruct-v0.2",
        help="You can try Granite if hosted: ibm-granite/granite-3.2-8b-instruct"
    )

    # Smart Pot
    st.markdown("---")
    sp = st.session_state.smart_pot
    sp["mode"] = st.selectbox("Smart Pot Mode", ["With Transaction", "Without Transaction"], index=0)
    sp["round_multiple"] = st.select_slider("Round multiple", [5,10,20,50,100], value=sp["round_multiple"])
    sp["goal_amount"] = st.number_input("Goal (₹)", min_value=0.0, step=100.0, value=float(sp["goal_amount"]))
    if st.button("Reset Pot (keep settings)"):
        sp["savings_total"] = 0.0
        sp["history"].clear()
        st.success("Smart Pot totals reset ✅")

    # Embedding status
    st.markdown("---")
    st.caption(f"Embeddings: {'✅ loaded' if EMBED_OK else '⚠️ not available (transformers/torch missing)'}")

    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================
st.markdown(
    """
    <div class="co-card">
      <h1>Coinie 🪙 <span class="co-badge">Frosted UI</span></h1>
      <p>Personalized, demographic-aware guidance for <span class="co-underline">savings</span>,
      <span class="co-underline">taxes</span>, and <span class="co-underline">investments</span>.
      Smart Pot rounds purchases to grow savings automatically.</p>
    </div>
    """, unsafe_allow_html=True
)

# ==============================
# NAV TABS
# ==============================
tabs = st.tabs(["🏠 Home", "🧠 NLU Analysis", "💬 Q&A", "📊 Budget Summary", "💡 Spending Insights"])

# ---------------- HOME ----------------
with tabs[0]:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("### Welcome to Coinie")
        st.markdown(
            "- **Smart Pot**: round every spend to build savings\n"
            "- **Demographic-aware** tone (Student/Professional)\n"
            "- **Q&A**, **Budget Summary**, **Spending Insights**\n"
            "- Optional **local embeddings** for semantic features\n"
        )
        st.info("Disclaimer: Educational information only — not financial, legal, or tax advice.")
    with col2:
        st.markdown("### Smart Pot")
        total, goal, pct = pot_progress()
        st.progress(pct/100)
        st.metric("Saved so far", f"₹ {total:,.2f}")
        st.caption(f"Goal: ₹ {goal:,.2f} • {pct}% complete")
        st.success(["","",*[""]][0] or f"{['',''][0]}")  # keep layout calm
        st.success(f"{WITTY[int(time.time()) % len(WITTY)]}")

    st.markdown("### Add a Transaction")
    with st.form("txn_form"):
        c1, c2, c3 = st.columns([1,1,2])
        amount = c1.number_input("Amount (₹)", min_value=0.0, step=10.0, value=56.0)
        multiple = c2.select_slider("Round to", [5,10,20,50,100], value=st.session_state.smart_pot["round_multiple"])
        ask_approve = c3.selectbox("Approve round-up now?", ["Yes","No"], index=0)
        submitted = st.form_submit_button("Add Transaction")
        if submitted:
            if ask_approve == "Yes" and st.session_state.smart_pot["mode"] in ("With Transaction","Without Transaction"):
                ent = add_pot_entry("purchase", amount, multiple, note="User-approved round-up")
                st.success(f"Rounded up to next ₹{multiple}. Saved ₹{ent['contribution']:.2f} 🪙")
            else:
                st.warning("Round-up not approved — no contribution added.")

    if st.session_state.smart_pot["mode"] == "Without Transaction":
        if st.button("Add Daily Contribution (No Transaction)"):
            ent = add_daily_without_txn(st.session_state.smart_pot["round_multiple"])
            st.success(f"Daily auto-save: ₹{ent['contribution']:.2f} added 🪙")

    with st.expander("Smart Pot History"):
        hist = st.session_state.smart_pot["history"]
        if not hist:
            st.caption("No entries yet.")
        else:
            st.write(hist)

# ---------------- NLU ANALYSIS ----------------
with tabs[1]:
    st.markdown("### Quick NLU (sentiment + keywords). Watson NLU can be integrated later.")
    text = st.text_area("Paste any finance text:", height=160)
    run_sent = st.toggle("Sentiment", True)
    run_kw = st.toggle("Keywords", True)

    if st.button("Analyze Text"):
        if not text.strip():
            st.warning("Please paste some text.")
        else:
            result = {}
            if run_sent:
                if not st.session_state.get("hf_warned", False) and not os.getenv("HF_API_KEY") and not st.session_state.get("hf_token_ui"):
                    st.session_state["hf_warned"] = True
                result["sentiment"] = simple_sentiment_hf(text, hf_token)
            if run_kw:
                result["keywords"] = naive_keywords(text, 10)

            # Optional semantic similarity demo if embeddings available
            if EMBED_OK:
                tips = [
                    "Cut discretionary food delivery by 20%.",
                    "Automate saving right after salary credit.",
                    "Use 50/30/20 as a baseline and adjust.",
                    "Start a small SIP for long-term goals.",
                ]
                vecs = embed_texts([text] + tips)
                base = vecs[0]
                sims = [(t, round(cosine(base, v), 3)) for t, v in zip(tips, vecs[1:])]
                sims.sort(key=lambda x: x[1], reverse=True)
                result["similar_tips_by_embedding"] = sims

            st.markdown("#### Results")
            st.write(result)

# ---------------- Q&A ----------------
with tabs[2]:
    st.markdown("### Ask Coinie")
    user_type = st.selectbox("User Type", ["Student","Professional"], index=0)
    q = st.text_area("Your question", placeholder="“How do I start an emergency fund while paying a student loan?”", height=120)

    if st.button("Get Answer"):
        if not q.strip():
            st.warning("Please enter a question.")
        elif not hf_token:
            st.error("Add your Hugging Face token in the sidebar.")
        else:
            prompt = prompt_qa(user_type, q)
            with st.spinner("Thinking..."):
                out = hf_generate(prompt, model_id, hf_token, max_new_tokens=280, temperature=0.35)
            st.session_state.chat_hist.append({"role":"user","content":q})
            st.session_state.chat_hist.append({"role":"assistant","content":out})
            st.markdown("#### Coinie says")
            st.write(out)

    if st.session_state.chat_hist:
        st.markdown("#### Conversation History")
        for m in st.session_state.chat_hist[-10:]:
            who = "👤" if m["role"]=="user" else "🤖"
            st.markdown(f"**{who} {m['role'].capitalize()}:** {m['content']}")

# ---------------- BUDGET SUMMARY ----------------
with tabs[3]:
    st.markdown("### Budget Summary")
    c1, c2 = st.columns(2)
    with c1:
        profile = st.selectbox("Profile", ["Student","Professional"], index=0)
        income = st.number_input("Monthly Income (₹)", min_value=0.0, step=1000.0, value=30000.0)
    with c2:
        default = {"Rent/Housing": 12000, "Food & Groceries": 6000, "Transport": 2500, "Subscriptions": 800, "Other": 2500}
        exp_json = st.text_area("Expenses JSON", value=json.dumps(default, indent=2), height=180)

    if st.button("Generate Summary"):
        try:
            cats = json.loads(exp_json)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            cats = None

        if cats:
            total_exp = sum(float(v) for v in cats.values() if isinstance(v,(int,float)))
            surplus = income - total_exp
            savings_rate = 0 if income<=0 else round(max(0.0, surplus)/income*100, 1)

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Expenses", f"₹ {total_exp:,.2f}")
            m2.metric("Surplus", f"₹ {surplus:,.2f}")
            m3.metric("Savings Rate", f"{savings_rate}%")

            if hf_token:
                prompt = prompt_budget(profile, income, cats)
                with st.spinner("Drafting narrative..."):
                    text = hf_generate(prompt, model_id, hf_token, max_new_tokens=320, temperature=0.35)
                st.markdown("#### Summary")
                st.write(text)
            else:
                st.info("Add a Hugging Face token for a narrative summary.")
            st.caption("Educational information only — not financial advice.")

# ---------------- SPENDING INSIGHTS ----------------
with tabs[4]:
    st.markdown("### Spending Insights")
    st.caption("Enter categories and monthly amounts. Coinie highlights optimizations.")
    preset = {"Rent/Housing": 15000, "Food & Groceries": 7000, "Transport": 3500, "Coffee": 1200, "Entertainment": 1500, "Other": 2500}
    rows = st.data_editor(
        [{"Category": k, "Amount": v} for k,v in preset.items()],
        num_rows="dynamic",
        key="spend_table"
    )

    if st.button("Analyze Spending"):
        cats = {}
        try:
            for r in rows:
                k = str(r.get("Category","")).strip()
                v = float(r.get("Amount",0))
                if k and v>=0:
                    cats[k]=v
        except Exception:
            st.error("Please ensure all amounts are numbers.")
            cats = {}

        if cats:
            total = sum(cats.values())
            top3 = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)[:3]
            st.markdown("#### Quick Heuristics")
            st.write(f"Total monthly spend: ₹ {total:,.2f}")
            for name, amt in top3:
                pct = 0 if total==0 else (amt/total)*100
                st.write(f"- **{name}**: ₹ {amt:,.2f} ({pct:.1f}%)")

            if hf_token:
                prompt = prompt_insights("Professional", cats)
                with st.spinner("Generating suggestions..."):
                    out = hf_generate(prompt, model_id, hf_token, max_new_tokens=320, temperature=0.35)
                st.markdown("#### Suggestions")
                st.write(out)
            else:
                st.info("Add a Hugging Face token for deeper suggestions.")
            st.caption("Educational information only — not financial advice.")
        else:
            st.warning("Add at least one category with an amount.")
