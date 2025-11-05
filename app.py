# app.py
# Streamlit + Gemini 샘플 챗봇 (single-file). 핵심 주석만 포함.
import streamlit as st
import requests
import time
import csv
import io
import pandas as pd
from datetime import datetime
import uuid

st.set_page_config(page_title="폐의약품 고객응대 챗봇", layout="wide")

# ---------- 설정 UI ----------
st.title("폐의약품 고객응대 챗봇 (Streamlit + Gemini)")

with st.expander("설정"):
    model_name = st.selectbox("모델 선택", options=["gemini-2.0-flash"], index=0)
    show_csv_option = st.checkbox("자동으로 대화 기록을 CSV에 저장", value=False)
    max_retries = st.number_input("429 재시도 최대 횟수", min_value=1, max_value=10, value=5)
    backoff_base = st.number_input("재시도 기본 지연(초)", min_value=1.0, max_value=10.0, value=1.0, step=0.5)

# API 키 (st.secrets 우선, 없으면 임시 입력 UI 제공)
GEMINI_API_KEY = None
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
else:
    st.warning("GEMINI_API_KEY가 st.secrets에 없습니다. 임시로 입력하면 세션에만 사용됩니다.")
    temp_key = st.text_input("임시 GEMINI API KEY (권장하지 않음 — st.secrets 사용 권장)", type="password")
    if temp_key:
        GEMINI_API_KEY = temp_key

if not GEMINI_API_KEY:
    st.error("API 키가 필요합니다. st.secrets['GEMINI_API_KEY']에 설정하거나 임시 키를 입력하세요.")
    st.stop()

# ---------- 세션 상태 초기화 ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "conversation_start" not in st.session_state:
    st.session_state["conversation_start"] = datetime.utcnow().isoformat()

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 친절하고 공감하는 고객응대 챗봇입니다. 주요 목적은 '폐의약품을 버리려는 고객'에게 안전한 폐기방법을 안내하고, 고객 불편을 구체적으로 정리하여 담당자에게 전달하는 것입니다.
응답은 정중하고 공감어린 말투로 하세요. 사용자가 제공한 정보를 (무엇이/언제/어디서/어떻게) 형태로 정리해 안내하고, 담당자 확인 후 회신을 위해 이메일 주소를 요청하세요. 사용자가 연락 제공을 원치 않으면: "죄송하지만, 연락처 정보를받지못하여담당자의검토내용을받으실수없어요."라고 정중히 안내하세요.
"""

if not any(m["role"] == "system" for m in st.session_state["messages"]):
    st.session_state["messages"].append({"role": "system", "text": SYSTEM_PROMPT, "time": datetime.utcnow().isoformat()})

col1, col2 = st.columns([3,1])

with col1:
    st.subheader("대화")
    st.caption(f"세션 ID: {st.session_state['session_id']}  •  모델: {model_name}")

    def render_history():
        for msg in st.session_state["messages"]:
            if msg["role"] == "system":
                continue
            when = msg.get("time", "")
            if msg["role"] == "user":
                st.markdown(f"**사용자:** {msg['text']}  \n*{when}*")
            elif msg["role"] == "assistant":
                st.markdown(f"**챗봇:** {msg['text']}  \n*{when}*")
    render_history()

    user_input = st.text_area("메시지 입력", height=120, key="user_input")
    col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
    with col_btn1:
        send = st.button("전송")
    with col_btn2:
        reset_conv = st.button("대화 초기화")
    with col_btn3:
        download_log = st.button("로그 다운로드 (CSV)")

with col2:
    st.subheader("도구")
    st.markdown("- 자동 CSV 저장: " + ("ON" if show_csv_option else "OFF"))
    st.markdown(f"- 최근 대화 길이: {len(st.session_state['messages'])}")
    st.markdown(f"- 대화 시작: {st.session_state['conversation_start']}")
    st.markdown("---")
    st.markdown("**최근 정리 예시(요청사항)**")
    st.code("무엇: 약품명/상태\n언제: 날짜/시간\n어디서: 구매처/보관장소\n어떻게: 현재상태/처분 시도 내역")
    st.markdown("---")
    st.markdown("사용자 개인정보는 최소한으로 수집하세요.")

API_URL_TEMPLATE = "https://generative.googleapis.com/v1beta2/models/{model}:generate"

def call_gemini_with_retry(prompt_text, context_messages, retries=5, backoff=1.0):
    url = API_URL_TEMPLATE.format(model=model_name)
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
    combined = ""
    for m in context_messages[-20:]:
        role = m["role"]
        combined += f"[{role.upper()}] {m['text']}\n"
    combined += f"[USER] {prompt_text}\n"

    payload = {"prompt": {"text": combined}, "temperature": 0.2, "candidate_count": 1, "max_output_tokens": 1024}

    attempt = 0
    while attempt <= retries:
        attempt += 1
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            j = resp.json()
            text = ""
            if "candidates" in j and isinstance(j["candidates"], list) and len(j["candidates"])>0:
                text = j["candidates"][0].get("output", {}).get("content", "") or j["candidates"][0].get("content", "")
            if not text:
                text = j.get("output", {}).get("text", "") or j.get("text", "")
            if not text:
                text = str(j)
            return text, j
        elif resp.status_code == 429:
            time.sleep(backoff * (2 ** (attempt-1)))
        else:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            return f"[ERROR] API 응답 오류: {resp.status_code} - {err}", None
    return "[RATE_LIMIT] 429 반복 발생", None

CSV_PATH = "/tmp/chatbot_logs.csv"
def append_to_csv(session_id, messages):
    rows = [{"session_id": session_id, "role": m["role"], "text": m["text"], "time": m.get("time", "")} for m in messages]
    df = pd.DataFrame(rows)
    try:
        df.to_csv(CSV_PATH, mode="a", header=not pd.io.common.file_exists(CSV_PATH), index=False, encoding="utf-8-sig")
    except Exception:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

def save_message(role, text):
    st.session_state["messages"].append({"role": role, "text": text, "time": datetime.utcnow().isoformat()})

if reset_conv:
    st.session_state["messages"] = [m for m in st.session_state["messages"] if m["role"]=="system"]
    st.session_state["conversation_start"] = datetime.utcnow().isoformat()
    st.success("대화를 초기화했습니다.")
    st.experimental_rerun()

if send and user_input.strip():
    save_message("user", user_input.strip())
    assistant_text, raw = call_gemini_with_retry(user_input.strip(), st.session_state["messages"], retries=int(max_retries), backoff=float(backoff_base))

    if assistant_text == "[RATE_LIMIT] 429 반복 발생":
        all_msgs = st.session_state["messages"]
        non_sys = [m for m in all_msgs if m["role"]!="system"]
        keep = non_sys[-12:] if len(non_sys) >= 12 else non_sys
        st.session_state["messages"] = [m for m in st.session_state["messages"] if m["role"]=="system"] + keep
        err_msg = "서비스가 일시적으로 혼잡합니다. 최근 대화 일부만 유지한 뒤 새롭게 시작합니다. 불편을 드려 죄송합니다."
        save_message("assistant", err_msg)
        st.error("429(요청량 제한) 문제가 반복 발생했습니다.")
    elif assistant_text.startswith("[ERROR]"):
        save_message("assistant", assistant_text)
        st.error("API 호출 중 오류가 발생했습니다.")
    else:
        save_message("assistant", assistant_text)

    if show_csv_option:
        try:
            append_to_csv(st.session_state["session_id"], st.session_state["messages"][-2:])
        except Exception as e:
            st.warning(f"CSV 저장 실패: {e}")

    st.rerun()

if download_log:
    rows = [[st.session_state["session_id"], m["role"], m["text"], m.get("time","")] for m in st.session_state["messages"]]
    bio = io.StringIO()
    writer = csv.writer(bio)
    writer.writerow(["session_id","role","text","time"])
    writer.writerows(rows)
    st.download_button("다운로드: 대화로그.csv", data=bio.getvalue(), file_name=f"chatlog_{st.session_state['session_id']}.csv", mime="text/csv")

st.markdown("---")
st.caption("개인정보 수집은 최소화하세요. 실제 배포 전엔 인증 방식을 보안 정책에 맞게 구현하세요.")
