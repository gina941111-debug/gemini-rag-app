import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from streamlit_mic_recorder import mic_recorder

# ===============================
# 1. Page config
# ===============================
st.set_page_config(
    page_title="Gemini ChatGPT Style Bot",
    layout="centered"
)
st.title("💬 Gemini 多模態機器人")

# ===============================
# 2. Gemini client
# ===============================
def setup_gemini_client():
    try:
        return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception as e:
        st.error(f"Gemini 初始化失敗：{e}")
        return None

client = setup_gemini_client()

# ===============================
# 3. Session state init
# ===============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "parts": ["你好！可以傳圖片、語音或文字提問 😊"]}
    ]

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "speech_buffer" not in st.session_state:
    st.session_state.speech_buffer = None

if "upload_counter" not in st.session_state:
    st.session_state.upload_counter = 0

if "clear_text" not in st.session_state:
    st.session_state.clear_text = False

if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

# ===============================
# 4. Sidebar – model
# ===============================
st.sidebar.markdown("## 🤖 Gemini 模型")
MODEL_OPTIONS = {
    "Gemini 2.5 Flash（穩定）": "gemini-2.5-flash",
    "Gemini 3.5 Flash": "gemini-3.5-flash",
    "Gemini 3.5 Pro": "gemini-3.5-pro",
}
model_name = st.sidebar.selectbox(
    "選擇模型",
    list(MODEL_OPTIONS.keys()),
    index=0
)
selected_model = MODEL_OPTIONS[model_name]

# ===============================
# 5. Chat history
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        for p in msg["parts"]:
            if isinstance(p, str):
                st.markdown(p)
            else:
                st.image(p, use_column_width=True)

# ===============================
# 6. ChatGPT-style input bar
# ===============================
st.markdown("---")

# 第一排：文字輸入 + 送出
col_text, col_send = st.columns([7, 1])

with col_text:
    default = "" if st.session_state.clear_text else st.session_state.get("chat_text", "")
    text_input = st.text_input(
        "輸入訊息",
        value=default,
        placeholder="輸入訊息…",
        label_visibility="collapsed",
        key="chat_text"
    )
    # 畫完這一輪後關掉清空旗標
    st.session_state.clear_text = False

with col_send:
    send_clicked = st.button("➤", use_container_width=True)

# 第二排：📎 上傳圖片 + 麥克風
col_img, col_mic = st.columns([1, 1])

with col_img:
    if st.button("📎 上傳圖片", use_container_width=True):
        st.session_state.show_uploader = not st.session_state.show_uploader

    if st.session_state.show_uploader:
        uploaded_file = st.file_uploader(
            "上傳圖片",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
            key=f"image_uploader_{st.session_state.upload_counter}"
        )
        if uploaded_file:
            st.session_state.uploaded_image = Image.open(uploaded_file)

with col_mic:
    mic_result = mic_recorder(
        start_prompt="🎙️ 開始錄音",
        stop_prompt="⏹️ 停止錄音",
        just_once=True,
        key="mic_recorder_main"
    )

# ===============================
# 7. Speech → Text (Gemini STT)
# ===============================
if mic_result and mic_result.get("bytes") and client:
    with st.spinner("🎧 語音轉文字中..."):
        audio_part = types.Part(
            inline_data=types.Blob(
                mime_type="audio/mp4",
                data=mic_result["bytes"]
            )
        )

        stt = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="請將這段語音完整轉成繁體中文，只輸出文字"
                        ),
                        audio_part
                    ]
                )
            ]
        )

        st.session_state.speech_buffer = stt.text
        st.success(f"🎧 {stt.text}")

# ===============================
# 8. Decide final prompt
# ===============================
final_prompt = None

if st.session_state.speech_buffer:
    final_prompt = st.session_state.speech_buffer
    st.session_state.speech_buffer = None
    st.session_state.clear_text = True          # 用語音時也清空輸入框
elif send_clicked and st.session_state.get("chat_text"):
    final_prompt = st.session_state.chat_text
    st.session_state.clear_text = True          # 文字送出後清空

# ===============================
# 9. Send to Gemini chat
# ===============================
if final_prompt and client:
    # 準備 user parts（可能含圖片）
    user_parts = []
    if st.session_state.uploaded_image:
        user_parts.append(st.session_state.uploaded_image)
    user_parts.append(final_prompt)

    # 畫出 user 訊息
    with st.chat_message("user"):
        for p in user_parts:
            if isinstance(p, str):
                st.markdown(p)
            else:
                st.image(p, use_column_width=True)

    # 存到歷史
    st.session_state.messages.append(
        {"role": "user", "parts": user_parts}
    )

    # 準備 Gemini contents（歷史對話）
    contents = []
    for m in st.session_state.messages:
        role = "model" if m["role"] == "assistant" else "user"
        parts = []
        for p in m["parts"]:
            if isinstance(p, str):
                parts.append(types.Part(text=p))
            else:
                buf = BytesIO()
                p.save(buf, format="PNG")
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=buf.getvalue()
                        )
                    )
                )
        contents.append(types.Content(role=role, parts=parts))

    # 呼叫 Gemini
    with st.chat_message("assistant"):
        try:
            with st.spinner("🤖 Gemini 思考中..."):
                response = client.models.generate_content(
                    model=selected_model,
                    contents=contents
                )
                answer = response.text
        except Exception as e:
            answer = f"❌ 模型呼叫失敗：{e}"
        st.markdown(answer)

    # 存回應
    st.session_state.messages.append(
        {"role": "assistant", "parts": [answer]}
    )

    # 🔑 清一次性狀態
    st.session_state.uploaded_image = None
    st.session_state.upload_counter += 1        # 讓下一輪 file_uploader 產生新 key
