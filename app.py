import streamlit as st
from google import genai
from google.genai import types

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from PIL import Image
from io import BytesIO
import base64
from streamlit_mic_recorder import mic_recorder

import fitz
import docx
import sqlite3
from datetime import datetime, timezone
import json


# ===============================
# 1. Page config
# ===============================
st.set_page_config(
    page_title="Gemini ChatGPT Style Bot + RAG",
    page_icon="🦦",
    layout="centered",
)
st.title("💬 Gemini 多模態機器人（RAG）")


# ===============================
# 2. Utils
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
# DB：永久記憶 (SQLite)
# ===============================
DB_PATH = "chat_memory.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            messages_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_memory(messages, mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO chat_memory (created_at, mode, messages_json)
        VALUES (?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            mode,
            json.dumps(messages, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def load_all_memory():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, created_at, mode FROM chat_memory ORDER BY id DESC"
    )
    rows = c.fetchall()
    conn.close()
    return rows


def load_memory_by_id(memory_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, created_at, mode, messages_json FROM chat_memory WHERE id = ?",
        (memory_id,),
    )
    row = c.fetchone()
    conn.close()
    return row


def delete_memory_by_id(memory_id: int):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_memory WHERE id = ?", (memory_id,))
    conn.commit()
    conn.close()


def delete_all_memory():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM chat_memory")
    conn.commit()
    conn.close()


# 初始化 DB
init_db()


# ===============================
# 3. Session state
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
# 4. Gemini client / LLM / Embeddings
#    （使用者從前端貼 API key）
# ===============================
st.sidebar.markdown("## 🔑 API Key 設定")
api_key_input = st.sidebar.text_input(
    "貼上你的 Gemini API Key",
    type="password",
)

if api_key_input:
    st.session_state["user_api_key"] = api_key_input

user_api_key = st.session_state.get("user_api_key")

if not user_api_key:
    st.sidebar.warning("請先貼上 Gemini API Key 才能開始使用")


def setup_gemini_client(api_key: str | None):
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        st.error(e)
        return None


def setup_llm(model_name: str, api_key: str | None):
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0.7,
    )


def setup_embeddings(api_key: str | None):
    if not api_key:
        return None
    return GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=api_key,
    )


client = setup_gemini_client(user_api_key)


# ===============================
# 5. Sidebar（模型 / 檔案 / 模式 / 記憶）
# ===============================
st.sidebar.markdown("## 🤖 模型")
MODEL_OPTIONS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 1.5": "gemini-robotics-er-1.5-preview",
}
selected_model_name = st.sidebar.selectbox("選擇模型", MODEL_OPTIONS.keys())
model_name = MODEL_OPTIONS[selected_model_name]

model = setup_llm(model_name, user_api_key)
embeddings = setup_embeddings(user_api_key)

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

st.sidebar.markdown("## 💾 對話記憶")

saved_list = load_all_memory()
if saved_list:
    options = {f"{r[0]} | {r[1][:19]} | {r[2]}": r[0] for r in saved_list}
    selected_label = st.sidebar.selectbox("已儲存對話", list(options.keys()))
    selected_id = options[selected_label]

    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("載入對話", key="load_memory"):
            row = load_memory_by_id(selected_id)
            if row:
                _, created_at, saved_mode, messages_json = row
                st.session_state.messages = json.loads(messages_json)
                mode = saved_mode
                st.sidebar.success(f"已載入對話 ID {selected_id}")
    with c2:
        if st.button("刪除此對話", key="delete_memory"):
            delete_memory_by_id(selected_id)
            st.sidebar.warning(f"已刪除對話 ID {selected_id}")
            st.rerun()

    if st.sidebar.button("🧨 刪除全部記憶", key="delete_all_memory"):
        delete_all_memory()
        st.sidebar.warning("已刪除全部儲存對話")
        st.rerun()
else:
    st.sidebar.info("目前沒有已儲存的對話")

if st.sidebar.button("💾 儲存目前對話", key="save_memory_now"):
    if st.session_state.get("messages"):
        save_memory(st.session_state.messages, mode)
        st.sidebar.success("已將目前對話永久儲存到資料庫")
        st.rerun()
    else:
        st.sidebar.warning("目前沒有對話可以儲存")


# ===============================
# ⭐ RAG Reset
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

if st.sidebar.button("🗑️ 清除上傳資料"):
    st.session_state.doc_vectorstore = None
    st.session_state.docs_loaded = False
    st.sidebar.success("資料已清除")


# ===============================
# 如果沒有 API key 或沒有模型，直接停止
# ===============================
if not user_api_key or not model:
    st.stop()


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
# 8. Input row（Enter 不送出）
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


def on_send():
    text = st.session_state.get("multi_enter_input", "").strip()
    if text:
        st.session_state["last_submitted_text"] = text
    st.session_state["multi_enter_input"] = ""


user_text = st.text_area(
    "輸入問題…（Enter 換行，按下方按鈕送出）",
    key="multi_enter_input",
    height=80,
)

st.button("送出", on_click=on_send)

prompt = st.session_state.pop("last_submitted_text", None)


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

    if st.session_state.uploaded_image:
        current_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(st.session_state.uploaded_image)
                    },
                },
                {"type": "text", "text": full_prompt},
            ]
        )
    else:
        current_message = HumanMessage(content=full_prompt)

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
