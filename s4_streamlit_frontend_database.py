import uuid
import streamlit as st
from l2_langgraph_database_backend import (
    chatbot, 
    model, 
    retrieve_all_threads, 
    save_thread_title, 
    retrieve_all_titles
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessageChunk


# Must be the first Streamlit call in the script
st.set_page_config(
    page_title="LangGraph ChatBot",
    initial_sidebar_state="expanded"
)

# ******************************************************* Utility Functions ************************************************************

def thread_id_generator() -> str:
    # Always a string, so it matches IDs loaded from the database
    return str(uuid.uuid4())


def new_chat():
    st.session_state['thread_id'] = None
    st.session_state['message_history'] = []


def add_thread(thread_id: str, title="New Chat"):
    thread_id = str(thread_id)
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    st.session_state['chat_titles'][thread_id] = title


def load_thread_chat(thread_id: str):
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    return state.values.get('message', [])


def fallback_title(user_message: str, max_len: int = 30) -> str:
    text = " ".join(user_message.split())
    if not text:
        return "New Chat"
    return text if len(text) <= max_len else text[:max_len].rstrip() + "..."


def generate_chat_title(user_message: str) -> str:
    title_prompt = [
        SystemMessage(content=(
            "You are a strict title extraction tool. Summarize the core topic of the user's input "
            "into a 3-5 word general title do not based on any specific given inputs. "
            "Do not add outside context, do not assume intentions, and do not "
            "add words not implied by the text. Return ONLY the plain text title. No quotes, no intro."
        )),
        HumanMessage(content=user_message)
    ]
    try:
        response = model.invoke(title_prompt)
        title = response.content.strip().strip('"\'')
        return title or fallback_title(user_message)
    except Exception:
        # Title generation must never block the actual chat
        return fallback_title(user_message)


def stream_ai_text(user_input: str, config: dict):
    """Yield only the assistant's text chunks coming from the chat node."""
    for chunk, metadata in chatbot.stream(
        {'message': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode='messages'
    ):
        if (
            metadata.get('langgraph_node') == 'chat_node'
            and isinstance(chunk, AIMessageChunk)
            and chunk.content
        ):
            yield chunk.content


# ******************************************************* Session State Setup **********************************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = retrieve_all_titles()


# ********************************************************* Sidebar UI *****************************************************************

st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('My Conversation')

# chat_threads is ordered oldest -> newest, so reverse for newest first
for thread_id in reversed(st.session_state['chat_threads']):
    title = st.session_state['chat_titles'].get(thread_id, "Untitled chat")
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_thread_chat(thread_id)

        temp_messages = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# ************************************************************* Main UI ****************************************************************

# Render the current chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    is_new_thread = False

    # Create a new thread only when the first message is sent
    if st.session_state['thread_id'] is None:
        thread_id = thread_id_generator()
        st.session_state['thread_id'] = thread_id
        title = generate_chat_title(user_input)
        save_thread_title(thread_id, title)
        add_thread(thread_id, title)
        is_new_thread = True

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    with st.chat_message('assistant'):
        ai_message = st.write_stream(stream_ai_text(user_input, CONFIG))

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if is_new_thread:
        st.rerun()