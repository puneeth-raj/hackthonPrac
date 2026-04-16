import streamlit as st
import sqlite3
from openai import OpenAI
from PyPDF2 import PdfReader

client = OpenAI()

# ---------------------------
# Database setup
# ---------------------------
conn = sqlite3.connect("chat.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS messages (
    role TEXT,
    content TEXT
)
""")
conn.commit()

def save_message(role, content):
    c.execute("INSERT INTO messages VALUES (?, ?)", (role, content))
    conn.commit()

def load_messages():
    c.execute("SELECT role, content FROM messages")
    return [{"role": r, "content": c} for r, c in c.fetchall()]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Advanced Chatbot")

st.title("💬 Smart Chatbot")

# Load chat history
if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

# ---------------------------
# Sidebar (RAG upload)
# ---------------------------
st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

rag_text = ""

if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        rag_text += page.extract_text()

    st.sidebar.success("Document loaded!")

# ---------------------------
# Display messages
# ---------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------
# Chat input
# ---------------------------
prompt = st.chat_input("Type your message...")

if prompt:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Typing indicator
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        # Add RAG context if available
        messages = st.session_state.messages.copy()

        if rag_text:
            messages.append({
                "role": "system",
                "content": f"Use this context to answer:\n{rag_text[:3000]}"
            })

        # Streaming response
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True
        )

        full_response = {"text": ""}

        def response_generator():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response["text"] += token
                    yield token
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token

        placeholder.empty()
        st.write_stream(response_generator)

    # Save assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    save_message("assistant", full_response)

# ---------------------------
# Clear chat
# ---------------------------
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.messages = []
    c.execute("DELETE FROM messages")
    conn.commit()