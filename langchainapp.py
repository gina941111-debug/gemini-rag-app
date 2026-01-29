import streamlit as st
from google import genai
from google.genai import types

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from PIL import Image
from io import BytesIO
import base64
from streamlit_mic_recorder import mic_recorder

import fitz
import docx

# ===============================
# 1. Page config
# ===============================
st.set_page_config(
    page_title="Gemini ChatGPT Style Bot + RAG",
    layout="centered",
)
st.title("💬 Gemini 多模態機器人（RAG）")

# ===============================
# 2. Gemini client（STT）
# ===============================
def setup_gemini_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(e)
        return None

client = setup_gemini_client()

# ===============================
# 3. LLM / Embeddings
# ===============================
def setup_llm(model):
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.7,
    )

def setup_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=st.secrets["GEMINI_API_KEY"],
    )

# ===============================
# 4. Utils
# ===============================
def encode_image(img: Image.Image):
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def extract_pdf(b):
    text = ""
    with fitz.open(stream=b, filetype="pdf") as d:
        for p in d:
            text += p.get_text()
    return text

def extract_docx(b):
    d = docx.Document(BytesIO(b))
    return "\n".join(p.text for p in d.paragraphs)
from langchain_core.messages import AIMessage

def build_chat_history(messages, current_human_message):
    """
    將 Streamlit messages 轉成 LangChain messages
    """
    chat = []
    for m in messages:
        if m["role"] == "user":
            chat.append(HumanMessage(content=m["parts"][0]))
        elif m["role"] == "assistant":
            chat.append(AIMessage(content=m["parts"][0]))

    chat.append(current_human_message)
    return chat

# ===============================
# 5. Session state
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "parts": ["你好！可以直接提問或上傳教材 😊"]}
    ]

for k, v in {
    "uploaded_image": None,
    "speech_buffer": None,
    "show_image_uploader": False,
    "upload_counter": 0,
    "doc_vectorstore": None,
    "docs_loaded": False,
}.items():
    st.session_state.setdefault(k, v)

# ===============================
# 6. Sidebar
# ===============================
st.sidebar.markdown("## 🤖 模型")
MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 1.5": "gemini-robotics-er-1.5-preview",
}
model = setup_llm(
    MODEL_OPTIONS[
        st.sidebar.selectbox("選擇模型", MODEL_OPTIONS.keys())
    ]
)
embeddings = setup_embeddings()

st.sidebar.markdown("## 📚 上傳檔案")
rag_files = st.sidebar.file_uploader(
    "TXT / MD / PDF / DOCX",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=True,
)

st.sidebar.markdown("## 🎓 學習模式")
mode = st.sidebar.selectbox(
    "選擇任務",
    ["一般聊天", "解釋/講解", "重點整理", "出小測驗"],
)

# ===============================
# ⭐ RAG Reset（關鍵）
# ===============================
if not rag_files:
    st.session_state.doc_vectorstore = None
    st.session_state.docs_loaded = False

# ===============================
# RAG 建索引
# ===============================
if rag_files and embeddings:
    all_text = ""
    for f in rag_files:
        raw = f.read()
        ext = f.name.split(".")[-1].lower()
        if ext in ["txt", "md"]:
            content = raw.decode("utf-8", errors="ignore")
        elif ext == "pdf":
            content = extract_pdf(raw)
        elif ext == "docx":
            content = extract_docx(raw)
        else:
            content = ""
        all_text += f"\n\n[檔名:{f.name}]\n{content}"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    docs = splitter.create_documents([all_text])
    st.session_state.doc_vectorstore = FAISS.from_documents(docs, embeddings)
    st.session_state.docs_loaded = True

# ===============================
# Sidebar 狀態顯示
# ===============================
if st.session_state.docs_loaded:
    st.sidebar.success("📚 教材模式啟用中")
else:
    st.sidebar.info("🤖 使用模型本身知識回答")

if st.sidebar.button("🗑️ 清除教材"):
    st.session_state.doc_vectorstore = None
    st.session_state.docs_loaded = False
    st.sidebar.success("教材已清除")

# ===============================
# 7. Chat history
# ===============================
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        for p in m["parts"]:
            if isinstance(p, str):
                st.markdown(p)
            else:
                st.image(p, use_container_width=True)

# ===============================
# 8. Input row
# ===============================
c1, c2 = st.columns([1, 1])

with c1:
    if st.button("🖼️", key=f"img_{st.session_state.upload_counter}"):
        st.session_state.show_image_uploader = True

with c2:
    mic = mic_recorder(
        start_prompt="🎙️",
        stop_prompt="⏹️",
        just_once=True,
        key=f"mic_{st.session_state.upload_counter}",
    )

if st.session_state.show_image_uploader:
    img = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key=f"imgf_{st.session_state.upload_counter}",
    )
    if img:
        st.session_state.uploaded_image = Image.open(img)
        st.session_state.show_image_uploader = False

prompt = st.chat_input("輸入問題…")

# ===============================
# 9. STT
# ===============================
if mic and mic.get("bytes") and client:
    audio = types.Part(
        inline_data=types.Blob(
            mime_type="audio/mp4",
            data=mic["bytes"],
        )
    )
    res = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="轉成繁體中文"), audio],
            )
        ],
    )
    st.session_state.speech_buffer = res.text

final_prompt = st.session_state.speech_buffer or prompt
st.session_state.speech_buffer = None

# ===============================
# 10. Gemini 回答（含記憶）
# ===============================
if final_prompt and model:

    instruction = {
        "一般聊天": "請正常回答問題。",
        "解釋/講解": "請用白話一步一步解釋。",
        "重點整理": "整理 3～7 個重點條列。",
        "出小測驗": "出 5 題選擇題，附答案與解釋。",
    }[mode]

    context = ""
    if st.session_state.doc_vectorstore:
        docs = st.session_state.doc_vectorstore.as_retriever(k=6).invoke(final_prompt)
        context = "\n\n".join(d.page_content for d in docs)

    full_prompt = f"""
{instruction}

【教材內容（若有）】
{context}

【使用者問題】
{final_prompt}
"""

    # 👉 本輪 HumanMessage
    if st.session_state.uploaded_image:
        current_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image(st.session_state.uploaded_image)},
                },
                {"type": "text", "text": full_prompt},
            ]
        )
    else:
        current_message = HumanMessage(content=full_prompt)

    # 👉 🔥 關鍵：把「整個聊天記憶」送進模型
    msgs = build_chat_history(
        st.session_state.messages,
        current_message
    )

    with st.chat_message("assistant"):
        with st.spinner("🤖 Gemini 思考中..."):
            answer = model.invoke(msgs).content
        st.markdown(answer)

    with st.chat_message("user"):
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image)
        st.markdown(final_prompt)

    st.session_state.messages += [
        {"role": "user", "parts": [final_prompt]},
        {"role": "assistant", "parts": [answer]},
    ]

    st.session_state.uploaded_image = None
    st.session_state.upload_counter += 1
