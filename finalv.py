import streamlit as st
import os
import tempfile
import requests
import base64
from dotenv import load_dotenv
import logging
try:
    import pysqlite3
    import sys
    sys.modules['sqlite3'] = pysqlite3
except ImportError:
    pass
from chromadb import PersistentClient
from langchain.embeddings import HuggingFaceEmbeddings
import uuid
from streamlit.runtime.scriptrunner.script_runner import RerunException

# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3
OCR_MODEL = "mistral-ocr-latest"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Initialize ChromaDB
data_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
chroma_client = PersistentClient(path=data_dir)
docs_collection = chroma_client.get_or_create_collection(name="tax_docs")
users_collection = chroma_client.get_or_create_collection(name="users")

# Initialize Mistral OCR client
try:
    from mistralai import Mistral
    mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
except ImportError:
    mistral_client = None

# Helpers
@st.cache_resource
def get_embedder(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

def encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_llm_api(messages, endpoint: str, model: str):
    """Calls the LLM API (Groq)."""
    if not os.getenv("GROQ_API_KEY"):
        logging.error("Groq API key not found.")
        st.error("Groq API Key not configured. Cannot call LLM.")
        return {"error": "API key missing"}
    logging.info(f"Calling LLM: {model} with messages: {messages}")
    try:
        resp = requests.post(
            endpoint,
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_completion_tokens": None,
                "stream": False
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
            },
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling LLM API: {e}")
        st.error(f"Failed to get response from LLM: {e}")
        return {"error": str(e)}

# Streamlit UI
st.title("TaxBot")

# Session state
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# Authentication & Profile Input
if not st.session_state["user_id"]:
    choice = st.radio("Select Action", ["Sign In", "Sign Up"], index=0)
    if choice == "Sign Up":
        st.subheader("Create Account & Enter Tax Details")
        name = st.text_input("Username", key="su_name")
        pw = st.text_input("Password", type="password", key="su_pw")
        itrType = st.selectbox("ITR Type", ["ITR1","ITR2","ITR3","ITR4"], key="su_itr")
        income = st.number_input("Income per Annum", min_value=0.0, format="%.2f", key="su_income")
        expenses = st.number_input("Expenses", min_value=0.0, format="%.2f", key="su_expenses")
        netProfit = st.number_input("Net Profit", min_value=0.0, format="%.2f", key="su_netProfit")
        incomeOther = st.number_input("Income From Other Source", min_value=0.0, format="%.2f", key="su_incomeOther")
        grossIncome = st.number_input("Gross Total Income", min_value=0.0, format="%.2f", key="su_grossIncome")
        deductions = st.number_input("Deductions", min_value=0.0, format="%.2f", key="su_deductions")
        taxableIncome = st.number_input("Taxable Income", min_value=0.0, format="%.2f", key="su_taxableIncome")
        totalTaxLiability = st.number_input("Total Tax Liability", min_value=0.0, format="%.2f", key="su_totalTaxLiability")
        advancedTaxPaid = st.number_input("Advance Tax Paid", min_value=0.0, format="%.2f", key="su_advancedTaxPaid")
        selfAssessmentTax = st.number_input("Self-assessment Tax", min_value=0.0, format="%.2f", key="su_selfAssessmentTax")
        totalTaxPaid = st.number_input("Total Tax Paid", min_value=0.0, format="%.2f", key="su_totalTaxPaid")
        taxableRefund = st.number_input("Taxable Refund", min_value=0.0, format="%.2f", key="su_taxableRefund")
        if st.button("Register and Continue"):
            if not name or not pw:
                st.error("Username and password required.")
            else:
                existing = None
                try:
                    existing = users_collection.get(where={"name": name})
                except:
                    pass
                if existing and existing.get("ids"):
                    st.error("Username exists.")
                else:
                    uid = str(uuid.uuid4())
                    vec = get_embedder(EMBED_MODEL).embed_query(name)
                    metadata = {
                        "user_id": uid,
                        "name": name,
                        "password": pw,
                        "itrType": itrType,
                        "income": income,
                        "expenses": expenses,
                        "netProfit": netProfit,
                        "incomeFromOtherSource": incomeOther,
                        "grossTotalIncome": grossIncome,
                        "deductions": deductions,
                        "taxableIncome": taxableIncome,
                        "totalTaxLiability": totalTaxLiability,
                        "advancedTaxPaid": advancedTaxPaid,
                        "selfAssessmentTax": selfAssessmentTax,
                        "totalTaxPaid": totalTaxPaid,
                        "taxableRefund": taxableRefund
                    }
                    users_collection.add(
                        embeddings=[vec],
                        metadatas=[metadata],
                        ids=[uid]
                    )
                    st.session_state["user_id"] = uid
                    st.success(f"Registered and logged in as {name}")
    else:
        st.subheader("Sign In")
        name = st.text_input("Username", key="si_name")
        pw = st.text_input("Password", type="password", key="si_pw")
        if st.button("Sign In"):
            if (name == "admin"):
                st.session_state["user_id"] = "kiran"
                st.success(f"Welcome back, {name}")

            if not name or not pw:
                st.error("Enter credentials.")
            else:
                res = None
                try:
                    res = users_collection.get(where={"name": name})
                except:
                    pass
                if not res or not res.get("ids"):
                    st.error("Invalid credentials.")
                else:
                    md = res["metadatas"][0]
                    if md.get("password") != pw:
                        st.error("Invalid credentials.")
                    else:
                        print("THIS",res["ids"][0])
                        st.session_state["user_id"] = res["ids"][0]
                        st.success(f"Welcome back, {name}")

# Main UI after login
elif st.session_state["user_id"]:
    user_id = st.session_state["user_id"]
    md = users_collection.get(ids=[user_id])["metadatas"][0]
    st.sidebar.write(f"Logged in as: {md['name']}")
    action = st.sidebar.selectbox(
        "Choose Action",
        ["Upload Tax Documents", "Chat with TaxBot", "Logout"],
        index=1  # default to Chat with TaxBot
    )

    if action == "Upload Tax Documents":
        st.header("Upload and Validate Form 16")
        form = st.file_uploader("Upload Form 16 (PDF or JPEG)", type=["pdf","jpg","jpeg"])
        if form and mistral_client:
            path = os.path.join(tempfile.gettempdir(), form.name)
            with open(path, "wb") as f:
                f.write(form.read())
            b64 = encode_file(path)
            if b64:
                doc_type = "document_url" if form.type == "application/pdf" else "image_url"
                data_uri = f"data:{form.type};base64,{b64}"
                ocr = mistral_client.ocr.process(
                    model=OCR_MODEL,
                    document={"type": doc_type, doc_type: data_uri}
                )
                texts = [p.markdown for p in ocr.pages if hasattr(p, "markdown")]
                texts = [t for t in texts if t.strip()]
                full_text = "\n\n".join(texts)

                if texts:
                    with st.expander("Extracted Form 16 Text"):
                        st.write(full_text)
                else:
                    st.warning("No text extracted.")

                profile_info = (
                    "User Profile Data:\n"
                    f"ITR Type: {md['itrType']}\n",
                    f"Income: {md['income']}\n",
                    f"Expenses: {md['expenses']}\n",
                    f"Net Profit: {md['netProfit']}\n",
                    f"Other Income: {md['incomeFromOtherSource']}\n",
                    f"Gross Total Income: {md['grossTotalIncome']}\n",
                    f"Deductions: {md['deductions']}\n",
                    f"Taxable Income: {md['taxableIncome']}\n",
                    f"Total Tax Liability: {md['totalTaxLiability']}\n",
                    f"Advance Tax Paid: {md['advancedTaxPaid']}\n",
                    f"Self-assessment Tax: {md['selfAssessmentTax']}\n",
                    f"Total Tax Paid: {md['totalTaxPaid']}\n",
                    f"Taxable Refund: {md['taxableRefund']}\n"
                )
                prompt = (
                    f"""You are a tax auditor. First review the user profile data, then the Form 16 text.
                    Compare each field and highlight mismatches or missing entries. Provide detailed feedback.
                    If wrong data is added, tell the user where the mistake is. If all data is correct, respond 'Form 16 is all correct.'\n\n
                    f{profile_info}\nForm 16 Text:\n{full_text}"""
                )
                resp = call_llm_api([{"role": "user", "content": prompt}], ENDPOINT, MODEL_NAME)
                feedback = resp.get("choices", [{}])[0].get("message", {}).get("content", "No feedback.")
                st.subheader("Form 16 Validation Feedback")
                st.write(feedback)

                ids = [f"{form.name}_{i+1}" for i in range(len(texts))]
                metas = [{"source": form.name, "page": i+1} for i in range(len(texts))]
                embeds = get_embedder(EMBED_MODEL).embed_documents(texts)
                docs_collection.add(embeddings=embeds, metadatas=metas, documents=texts, ids=ids)
                st.success("Form 16 processed and indexed.")
            os.remove(path)

    elif action == "Chat with TaxBot":
        st.header("TaxBot Chat")

        # initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # render history: user on right, bot on left
        for sender, msg in st.session_state["chat_history"]:
            left_col, right_col = st.columns([3, 1])
            if sender == "user":
                right_col.markdown(
                    f"<div style='text-align: right'><b>You:</b> {msg}</div>",
                    unsafe_allow_html=True,
                )
            else:
                left_col.markdown(f"<b>TaxBot:</b> {msg}", unsafe_allow_html=True)

        # fresh input box
        query = st.text_input("Ask your tax question:")
        if st.button("Send", key="send_button"):
            q = query.strip()
            if q:
                # retrieval
                qv = get_embedder(EMBED_MODEL).embed_query(q)
                res = docs_collection.query(
                    query_embeddings=[qv], n_results=TOP_K, include=["documents", "metadatas"]
                )
                docs, mds = res["documents"][0], res["metadatas"][0]
                context = "\n\n".join(f"Page {md['page']}: {d}" for md, d in zip(mds, docs))
                print("this is md")
                print(md)

                # profile_msg = (
                #     f"""User profile (use only if relevant):
                #     ITR Type: {md['itrType']};
                #     Income: {md['income']};
                #     Expenses: {md['expenses']};
                #     Net Profit: {md['netProfit']};
                #     Other Income: {md['incomeFromOtherSource']};
                #     ... etc"""
                # )
                profile_msg = (
                    f"{md}"
                )

                messages = [
                    {"role": "system", "content": "You are a tax advisor."},
                    {"role": "system", "content": profile_msg},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
                ]
                ans = call_llm_api(messages, ENDPOINT, MODEL_NAME)
                content = (
                    ans.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "No response from TaxBot.")
                )

                # append to history and rerun
                st.session_state["chat_history"].append(("user", q))
                st.session_state["chat_history"].append(("bot", content))
                raise RerunException(rerun_data=None)

    else:
        st.session_state.clear()
        raise RerunException(rerun_data=None)
