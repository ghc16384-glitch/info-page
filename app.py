import streamlit as st
import google.generativeai as genai
import time
import datetime
import pandas as pd
import io
import random
import string
from tenacity import retry, stop_after_attempt, wait_random_exponential

# --- 시스템 프롬프트 정의 ---
# 요청하신 프롬프트 내용을 여기에 적용합니다.
SYSTEM_PROMPT = """
1. 당신은 '폐의약품 올바르게 버리기'를 돕는 친절하고 상냥한 안내원입니다. 사용자의 질문에 항상 공감하며 긍정적인 말투로 응답하세요.
2. 사용자는 유통기한이 지났거나 사용하지 않는 약(폐의약품)의 폐기 방법을 문의할 것입니다.
3. **핵심 안내 (필수):** "폐의약품은 토양, 수질 오염을 유발할 수 있어 일반 쓰레기나 하수구, 변기에 버리시면 절대 안 됩니다. 반드시 가까운 **약국**이나 **보건소**에 비치된 **'폐의약품 전용 수거함'**에 가져다주셔야 합니다."라고 명확하게 첫 답변으로 안내하세요.
4. **상세 안내 (종류별):** 사용자가 약의 종류(알약, 물약, 연고 등)를 언급하거나 물어보면, 다음과 같이 종류별 분리배출 방법을 상세히 안내하세요.
    * **알약/캡슐:** 포장(PTP, 약병 등)은 분리수거하고, 알약만 모아서 한 봉투에 담아 수거함에 넣어주세요.
    * **물약/시럽:** 내용물이 새어 나오지 않게 병을 잘 잠근 후, 병 그대로 수거함에 넣어주세요. (절대 하수구에 버리지 마세요!)
    * **연고/안약/흡입기/스프레이 등 특수 형태:** 겉 종이 상자만 분리배출하고, 용기나 튜브는 그대로 수거함에 넣어주세요.
5. **위치 안내:** 사용자가 "수거함이 어디 있는지" 묻는다면, "대부분의 동네 약국이나 보건소에 비치되어 있습니다. 방문 전 전화를 해보시거나, 포털 지도 앱에서 '약국' 또는 '보건소'를 검색해 보시면 편리합니다."라고 안내하세요. (챗봇이 위치 정보를 직접 수집하거나 검색하지 않습니다.)
6. **마무리 인사:** 안내가 끝난 후, "더 궁금한 점 있으신가요?"라고 물어보고, 대화가 종료될 때는 "올바른 의약품 배출로 환경 보호에 동참해 주셔서 정말 감사합니다! 좋은 하루 보내세요."와 같은 긍정적인 인사로 마무리하세요.
"""

# --- API 호출 (429 재시도 로직 포함) ---
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_response_with_retry(model, chat_history):
    """
    API 호출을 시도하고, 429 에러 발생 시 재시도합니다.
    """
    try:
        # 시스템 프롬프트를 포함하여 대화 시작
        chat = model.start_chat(history=chat_history)
        # 마지막 사용자 메시지 전송
        response = chat.send_message(chat_history[-1]['parts'][0])
        return response.text
    except Exception as e:
        if "429" in str(e):
            st.warning("API 요청 한도에 도달했습니다. 잠시 후 재시도합니다...")
            raise e  # 재시도를 위해 예외를 다시 발생시킵니다.
        else:
            st.error(f"API 호출 중 오류 발생: {e}")
            return None

# --- 대화 내용 CSV 변환 ---
def convert_history_to_csv(history):
    """
    st.session_state.messages를 CSV 문자열로 변환합니다.
    """
    df = pd.DataFrame(history)
    df = df[df['role'] != 'system'] # 시스템 프롬프트는 제외
    df['time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['session_id'] = st.session_state.session_id
    
    # parts 컬럼의 리스트/딕셔너리 구조를 텍스트로 변환
    df['parts'] = df['parts'].apply(lambda x: x[0] if isinstance(x, list) and x else (x.get('text', '') if isinstance(x, dict) else x))
    
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()

# --- 메인 앱 실행 ---
def main():
    st.set_page_config(
        page_title="폐의약품 안내 챗봇",
        page_icon="💊"
    )

    st.title("💊 폐의약품 올바르게 버리기 안내 챗봇")

    # --- 1. API 키 관리 ---
    api_key = None
    try:
        # (권장) Streamlit Secrets에서 API 키 가져오기
        api_key = st.secrets.get('GEMINI_API_KEY')
    except Exception:
        pass # secrets가 없으면 무시

    if not api_key:
        with st.sidebar:
            st.warning("GEMINI_API_KEY가 secrets에 설정되지 않았습니다.")
            api_key = st.text_input("Gemini API 키를 입력하세요:", type="password")
    
    if not api_key:
        st.info("사이드바에서 Gemini API 키를 입력해주세요.")
        st.stop()
        
    # API 키 설정
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        st.error(f"API 키 설정에 실패했습니다: {e}")
        st.stop()

    # --- 2. 세션 상태 초기화 ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))

    if "model_name" not in st.session_state:
        st.session_state.model_name = "gemini-2.5-flash-preview-09-2025" # 기본 모델

    if "messages" not in st.session_state:
        # 시스템 프롬프트를 대화 기록의 첫 번째 요소로 추가
        st.session_state.messages = [{"role": "system", "parts": [SYSTEM_PROMPT]}]

    if "csv_log" not in st.session_state:
        st.session_state.csv_log = [] # CSV 기록용 리스트

    # --- 3. 사이드바 기능 ---
    with st.sidebar:
        st.header("챗봇 설정")
        
        # 모델 선택 (필요시 확장 가능)
        st.session_state.model_name = st.selectbox(
            "AI 모델 선택",
            ("gemini-2.5-flash-preview-09-2025", "gemini-pro"), # gemini-2.0-flash는 API 목록에 없어 2.5로 대체
            index=0
        )
        
        # 세션 ID 표시
        st.text_input("현재 세션 ID", st.session_state.session_id, disabled=True)

        # 대화 초기화 버튼
        if st.button("대화 초기화", type="primary"):
            st.session_state.messages = [{"role": "system", "parts": [SYSTEM_PROMPT]}]
            st.session_state.csv_log = []
            st.session_state.session_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            st.rerun()

        st.divider()
        
        # CSV 자동 기록 옵션 (체크박스)
        record_csv = st.checkbox("대화 내용 CSV로 자동 기록", value=True)
        
        # 로그 다운로드 버튼
        if st.session_state.csv_log:
            csv_data = convert_history_to_csv(st.session_state.csv_log)
            st.download_button(
                label="대화 기록 다운로드 (CSV)",
                data=csv_data,
                file_name=f"chat_log_{st.session_state.session_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # --- 4. 대화 히스토리 표시 ---
    # 시스템 프롬프트를 제외하고 사용자에게 표시
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["parts"][0])
        elif message["role"] == "model":
            # "핵심 안내" 부분 볼드 처리
            if "반드시 가까운 **약국**이나 **보건소**" in message["parts"][0]:
                 with st.chat_message("assistant"):
                    st.markdown(message["parts"][0], unsafe_allow_html=True)
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["parts"][0])

    # --- 5. 사용자 입력 처리 ---
    if prompt := st.chat_input("폐의약품 버리는 방법을 물어보세요..."):
        # 사용자 메시지 추가
        user_message = {"role": "user", "parts": [prompt]}
        st.session_state.messages.append(user_message)
        if record_csv:
            st.session_state.csv_log.append(user_message)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- 6. AI 응답 생성 ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("생각 중... 👩‍⚕️")
            
            try:
                # 모델 생성
                model = genai.GenerativeModel(
                    st.session_state.model_name,
                    # 시스템 프롬프트를 generation_config가 아닌 모델 초기화 시 전달
                    system_instruction=SYSTEM_PROMPT
                )
                
                # 히스토리 관리 (최근 6턴 = 12개 메시지 + 시스템 프롬프트 1개)
                # 시스템 프롬프트를 제외하고 최근 12개(6턴)를 선택
                recent_history = st.session_state.messages[1:] # 시스템 프롬프트 제외
                if len(recent_history) > 12:
                    recent_history = recent_history[-12:]
                
                # API 호출 (시스템 프롬프트는 모델에 설정되었으므로 히스토리에서는 제외하고 전송)
                full_response = generate_response_with_retry(model, recent_history)
                
                if full_response:
                    message_placeholder.markdown(full_response, unsafe_allow_html=True)
                    model_message = {"role": "model", "parts": [full_response]}
                    st.session_state.messages.append(model_message)
                    if record_csv:
                         st.session_state.csv_log.append(model_message)
                else:
                    message_placeholder.error("응답을 생성하지 못했습니다.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                # 오류 발생 시 마지막 사용자 메시지 제거
                if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                    st.session_state.messages.pop()
                if st.session_state.csv_log and st.session_state.csv_log[-1]["role"] == "user":
                    st.session_state.csv_log.pop()

if __name__ == "__main__":
    main()
