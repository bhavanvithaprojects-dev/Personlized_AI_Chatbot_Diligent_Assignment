import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Bhavanvitha's Personalized AI Chatbot",
    page_icon="💬",
    layout="centered"
)

# ----------------------------------
# UI Header
# ----------------------------------
st.markdown(
    """
    <h1 style='text-align:center;'>💬 Bhavanvitha AI Chatbot</h1>
    <p style='text-align:center;color:gray;'>
    Local AI Assistant powered by Ollama
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------------
# Session Memory
# ----------------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

SESSION_ID = "bhavanvitha_session"

# ----------------------------------
# Ollama LLM
# ----------------------------------
llm = ChatOllama(
    model="llama3",   # or mistral
    temperature=0.3
)

# ----------------------------------
# Prompt
# ----------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Bhavanvitha's personal AI assistant. "
        "Be friendly, professional, and helpful. "
        "Remember past conversations."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

base_chain = prompt | llm

# ----------------------------------
# Memory Handler
# ----------------------------------
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

chain = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ----------------------------------
# Show Chat History
# ----------------------------------
history = st.session_state.store.get(SESSION_ID)

if history:
    for msg in history.messages:
        with st.chat_message("assistant" if msg.type == "ai" else "user"):
            st.markdown(msg.content)

# ----------------------------------
# User Input
# ----------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": SESSION_ID}}
    )

    with st.chat_message("assistant"):
        st.markdown(response.content)

# ----------------------------------
# Sidebar
# ----------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🧹 Clear Chat"):
        st.session_state.store[SESSION_ID] = ChatMessageHistory()
        st.experimental_rerun()

    st.markdown("---")
    st.caption("© 2026 Bhavanvitha | Local GenAI Project")
