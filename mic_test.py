import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="Mic 測試", layout="centered")
st.title("🎙️ Streamlit 麥克風錄音測試")

st.markdown("點一次開始錄音，再點一次停止錄音。停止後下方會顯示錄音資料，並可播放聲音。")

# ===== 錄音元件 =====
audio = mic_recorder(
    start_prompt="點擊開始錄音",
    stop_prompt="點擊停止錄音",
    just_once=True,          # 錄完只回傳一次
    key="test_mic"
)

# ===== Debug：顯示回傳內容 =====
st.write("🔍 audio =", audio)

# ===== 若有錄到聲音，顯示播放按鈕 =====
if audio:
    if "bytes" in audio:
        st.success("錄音完成，可以播放。")
        st.audio(audio["bytes"], format="audio/wav")
    else:
        st.warning("有回傳資料，但不包含音訊 bytes。")
