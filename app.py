import re
from pathlib import Path

import streamlit as st
import joblib

st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="centered")

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "spam_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()

# ---------- ค่าตั้งต้น ----------
UNCERTAIN_LOW = 0.35    # ต่ำกว่านี้ = HAM
UNCERTAIN_HIGH = 0.65   # สูงกว่านี้ = SPAM (ระหว่างสองค่า = ไม่แน่ใจ)
MIN_WORDS = 2           # ต้องมีคำภาษาอังกฤษอย่างน้อยกี่คำ

EXAMPLES = {
    "🎁 Spam: Prize": "URGENT! You have won a 1 week FREE membership in our prize "
                      "draw. Text CLAIM to 81010 now. T&Cs apply",
    "💰 Spam: Cash": "Congratulations! You have won a free $1000 prize. "
                     "Call 09061701461 now to claim your reward!",
    "💬 Ham: Class": "Hey, are you coming to class today?",
    "🍜 Ham: Dinner": "Are you free tonight? Let's grab dinner near the "
                      "station around 7.",
}

# ---------- สไตล์ ----------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@400;500;600;700&display=swap');

.stApp p, .stApp label, .stApp li, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .hero, .result, .note {
    font-family: 'Prompt', sans-serif;
}
.stApp { background: linear-gradient(180deg, #f3f1ff 0%, #ffffff 40%); }
.block-container { padding-top: 1.5rem; max-width: 720px; }

.hero {
    background: linear-gradient(135deg, #6d5dfc 0%, #4f8cff 100%);
    border-radius: 24px; padding: 28px 22px; text-align: center; color: #fff;
    box-shadow: 0 12px 30px rgba(109, 93, 252, .28); margin-bottom: 22px;
}
.hero .emoji { font-size: 48px; line-height: 1; }
.hero h1 { color: #fff; margin: 8px 0 4px; font-size: 1.9rem; font-weight: 700; padding: 0; }
.hero p { color: #eef0ff; margin: 0 0 12px; font-size: .95rem; }
.badge {
    display: inline-block; background: rgba(255,255,255,.22); color: #fff;
    padding: 4px 14px; border-radius: 999px; font-size: .8rem; font-weight: 500;
}

.section-title { font-weight: 600; color: #1f2340; margin: 18px 0 6px; font-size: 1rem; }

.stTextArea textarea {
    border-radius: 16px !important; border: 2px solid #e2defc !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus { border-color: #6d5dfc !important; }
.stButton > button { border-radius: 14px; font-weight: 500; }
.stButton > button[kind="primary"] {
    padding: .7rem 1rem; font-size: 1.05rem;
    background: linear-gradient(135deg, #6d5dfc, #4f8cff); border: none;
}

.result {
    border-radius: 20px; padding: 22px; margin-top: 18px; background: #fff;
    box-shadow: 0 8px 24px rgba(31, 35, 64, .10); border-left: 8px solid;
}
.result.spam { border-color: #ef4444; }
.result.ham { border-color: #22c55e; }
.result.unsure { border-color: #f59e0b; }
.result .label { font-size: 1.6rem; font-weight: 700; margin: 0; }
.result.spam .label { color: #dc2626; }
.result.ham .label { color: #16a34a; }
.result.unsure .label { color: #d97706; }
.result .sub { color: #5b6080; margin: 4px 0 16px; font-size: .95rem; }

.gauge { position: relative; height: 14px; border-radius: 999px; }
.gauge-marker {
    position: absolute; top: -6px; width: 6px; height: 26px; border-radius: 4px;
    background: #1f2340; transform: translateX(-50%);
}
.gauge-labels {
    display: flex; justify-content: space-between; margin-top: 8px;
    font-size: .75rem; color: #7a7f9e; font-weight: 500;
}
.note { color: #7a7f9e; font-size: .8rem; text-align: center; margin-top: 26px; }
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- ฟังก์ชันช่วย ----------
def has_non_english_letters(text):
    return any(c.isalpha() and not c.isascii() for c in text)


def keep_english_only(text):
    cleaned = "".join(" " if (c.isalpha() and not c.isascii()) else c for c in text)
    return re.sub(r"\s+", " ", cleaned).strip()


def count_english_words(text):
    return len(re.findall(r"[A-Za-z]{2,}", text))


def set_example(text):
    st.session_state["msg"] = text


def render_result(spam_prob):
    if spam_prob > UNCERTAIN_HIGH:
        kind, label = "spam", "🚨 SPAM"
        sub = f"โอกาสเป็น Spam {spam_prob * 100:.1f}% - ข้อความนี้น่าสงสัย ระวังอย่ากดลิงก์หรือโทรกลับ"
    elif spam_prob < UNCERTAIN_LOW:
        kind, label = "ham", "✅ HAM"
        sub = f"โอกาสเป็นข้อความปกติ {(1 - spam_prob) * 100:.1f}%"
    else:
        kind, label = "unsure", "⚠️ ไม่แน่ใจ"
        sub = (f"โอกาสเป็น Spam {spam_prob * 100:.1f}% - ข้อความสั้นหรือกำกวม "
               "ลองพิมพ์ให้ยาวและสมจริงขึ้น")

    lo, hi = UNCERTAIN_LOW * 100, UNCERTAIN_HIGH * 100
    track = (f"linear-gradient(90deg, #22c55e 0%, #22c55e {lo}%, "
             f"#f59e0b {lo}%, #f59e0b {hi}%, #ef4444 {hi}%, #ef4444 100%)")
    st.markdown(
        f"""
        <div class="result {kind}">
          <p class="label">{label}</p>
          <p class="sub">{sub}</p>
          <div class="gauge" style="background:{track}">
            <div class="gauge-marker" style="left:{spam_prob * 100:.1f}%"></div>
          </div>
          <div class="gauge-labels"><span>HAM</span><span>ไม่แน่ใจ</span><span>SPAM</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------- หน้าเว็บ ----------
st.markdown(
    """
    <div class="hero">
      <div class="emoji">📱</div>
      <h1>SMS Spam Detector</h1>
      <p>ตรวจสอบข้อความ SMS ว่าเป็น Spam หรือไม่ ด้วย Machine Learning</p>
      <span class="badge">🇬🇧 English only</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.session_state.setdefault("msg", "")

st.markdown('<div class="section-title">✨ ลองตัวอย่าง</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, (name, text) in enumerate(EXAMPLES.items()):
    cols[i % 2].button(name, key=f"ex{i}", on_click=set_example, args=(text,),
                       use_container_width=True)

st.markdown('<div class="section-title">✍️ หรือพิมพ์ข้อความเอง (ภาษาอังกฤษ)</div>',
            unsafe_allow_html=True)
message = st.text_area("message", key="msg", height=130,
                       placeholder="Type or paste an English SMS here...",
                       label_visibility="collapsed")

if st.button("🔍 ตรวจสอบข้อความ", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("กรุณากรอกข้อความก่อน")
    else:
        text = keep_english_only(message)
        if count_english_words(text) < MIN_WORDS:
            st.error(
                "❌ ไม่พบข้อความภาษาอังกฤษที่เพียงพอ  \n"
                f"ระบบรองรับเฉพาะภาษาอังกฤษ และต้องมีอย่างน้อย {MIN_WORDS} คำ  \n"
                f"Please enter your message in English (at least {MIN_WORDS} words)."
            )
        else:
            if has_non_english_letters(message):
                st.info(
                    "ℹ️ พบตัวอักษรที่ไม่ใช่ภาษาอังกฤษ ระบบข้ามส่วนนั้นและทำนายจากส่วน"
                    f"ภาษาอังกฤษเท่านั้น  \nข้อความที่ใช้ทำนาย: `{text}`"
                )
            X = vectorizer.transform([text])
            spam_prob = float(model.predict_proba(X)[0][1])
            render_result(spam_prob)

with st.expander("ℹ️ เกี่ยวกับโมเดลนี้"):
    st.markdown(
        """
- **ข้อมูล:** UCI SMS Spam Collection (5,574 ข้อความภาษาอังกฤษ)
- **วิธีแปลงข้อความ:** TF-IDF
- **โมเดลที่ใช้:** Linear SVM (เลือกจากการเปรียบเทียบกับ Naive Bayes และ Logistic Regression)
- **ผลบนชุดทดสอบ:** Accuracy 98.3% | Precision 95.8% | Recall 91.3% | F1 93.5%
- **ข้อจำกัด:** รองรับภาษาอังกฤษเท่านั้น ข้อความสั้นมากอาจไม่แน่นอน และ spam
  รูปแบบใหม่ (เช่น ลิงก์หลอก) อาจจับไม่ได้
        """
    )

st.markdown('<div class="note">Made for a Machine Learning class project</div>',
            unsafe_allow_html=True)