import uuid
import streamlit as st
from l2_langgraph_database_backend import (
    chatbot,
    model_with_tools,
    retrieve_all_threads,
    retrieve_all_titles,
    save_thread_title,
    rename_thread_title,
    delete_thread_data,
)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AIMessageChunk


# Must be the first Streamlit call in the script
st.set_page_config(
    page_title="LangGraph ChatBot",
    initial_sidebar_state="expanded"
)


# ******************************************************* Utility Functions ************************************************************

def thread_id_generator() -> str:
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
    return state.values.get('messages', [])


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
        response = model_with_tools.invoke(title_prompt)
        title = response.content.strip().strip('"\'')
        return title or fallback_title(user_message)
    except Exception:
        return fallback_title(user_message)


def stream_reply(user_input: str, config: dict, placeholder) -> str:
    """Stream the assistant's answer, showing text only from turns that contain no tool call."""
    finished, current, step, is_tool = "", "", None, False

    for chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode='messages',
    ):
        if metadata.get('langgraph_node') != 'chat_node' or not isinstance(chunk, AIMessageChunk):
            continue

        # A new LLM call started: keep the previous one only if it was a plain answer
        if metadata.get('langgraph_step') != step:
            if not is_tool:
                finished += current
            current, is_tool, step = "", False, metadata.get('langgraph_step')

        current += chunk.text
        if chunk.tool_call_chunks:
            is_tool = True

        placeholder.markdown(finished if is_tool else finished + current)

    if not is_tool:
        finished += current
    placeholder.markdown(finished)
    return finished


# ---- Sidebar callbacks -------------------------------------------------------------------

def select_thread(thread_id: str):
    history = []
    for msg in load_thread_chat(thread_id):
        if isinstance(msg, HumanMessage):
            role = 'user'
        elif isinstance(msg, AIMessage) and msg.text and not msg.tool_calls:
            role = 'assistant'
        else:
            continue        # skip ToolMessages and tool-call turns
        history.append({'role': role, 'content': msg.text})

    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = history


def rename_chat(thread_id: str):
    new_title = st.session_state.get(f"rename_{thread_id}", "").strip()
    if new_title:
        rename_thread_title(thread_id, new_title)
        st.session_state['chat_titles'][thread_id] = new_title


def delete_chat(thread_id: str):
    delete_thread_data(thread_id)
    st.session_state['chat_threads'] = [t for t in st.session_state['chat_threads'] if t != thread_id]
    st.session_state['chat_titles'].pop(thread_id, None)
    if st.session_state['thread_id'] == thread_id:
        new_chat()


# ******************************************************* Session State Setup **********************************************************
# The database is read once per browser session, not on every rerun.

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

st.sidebar.button('New Chat', on_click=new_chat)

st.sidebar.header('My Conversation')

# chat_threads is ordered oldest -> newest, so reverse for newest first
for tid in reversed(st.session_state['chat_threads']):
    title = st.session_state['chat_titles'].get(tid, "Untitled chat")
    is_current = tid == st.session_state['thread_id']

    col_title, col_menu = st.sidebar.columns([5, 1])

    col_title.button(
        title,
        key=f"select_{tid}",
        on_click=select_thread,
        args=(tid,),
        use_container_width=True,
        type="primary" if is_current else "secondary",
    )

    with col_menu.popover("⋮"):
        st.text_input("Rename chat", value=title, key=f"rename_{tid}")
        st.button("Save name", key=f"save_{tid}", on_click=rename_chat, args=(tid,))
        st.button("Delete chat", key=f"delete_{tid}", on_click=delete_chat, args=(tid,))


# ************************************************************* Main UI ****************************************************************

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

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}, 'recursion_limit': 10}

    with st.chat_message('assistant'):
        placeholder = st.empty()
        ai_message = stream_reply(user_input, CONFIG, placeholder)

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if is_new_thread:
        st.rerun()