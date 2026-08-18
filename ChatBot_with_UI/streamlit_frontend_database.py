import uuid
import streamlit as st
from langgraph_database_backend import chatbot, model, retrieve_all_threads, save_thread_title, retrieve_all_titles
from langchain_core.messages import HumanMessage, SystemMessage


# ******************************************************* Utility Functions ************************************************************

def thread_id_generator():
    thread_id = uuid.uuid4()
    return thread_id


def new_chat():
    st.session_state['thread_id'] = None
    st.session_state['message_history'] = []


def add_thread(thread_id, title="New Chat"):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        st.session_state['chat_titles'][thread_id] = title


def load_thread_chat(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('message', [])


def generate_chat_title(user_message: str) -> str:
    title_prompt = [
    SystemMessage(content=(
                "You are a strict title extraction tool. Summarize the core topic of the user's input "
                "into a 3-5 word general title do not based on any specific given inpupts."
                "Do not add outside context, do not assume intentions, and do not "
                "add words not implied by the text. Return ONLY the plain text title. No quotes, no intro."
            )),        
            HumanMessage(content=user_message)
    ]
    response = model.invoke(title_prompt)
    return response.content.strip().strip('"\'')


# ******************************************************* Session State Setup **********************************************************

# st.session_state setup -> dict 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'chat_titles' not in st.session_state:
    st.session_state['chat_titles'] = retrieve_all_titles()

# add_thread(st.session_state['thread_id'])

# ********************************************************* Sidebar UI *****************************************************************
st.set_page_config(
    page_title="LangGraph ChatBot",
    initial_sidebar_state="expanded"
)

st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('My Conversation')

# st.sidebar.text(st.session_state['thread_id'])
for thread_id in reversed(st.session_state['chat_threads']):
    title = st.session_state['chat_titles'].get(thread_id, str(thread_id))
    if st.sidebar.button(title, key = str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_thread_chat(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})
            
        st.session_state['message_history'] = temp_messages


# *************************************************************Main UI ******************************************************************

# loading the past chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


# [{'role': 'user', 'content': 'Hi'}
#  {'role': 'assistent', 'content': 'Hello'}]

user_input = st.chat_input('Type here')

if user_input:

    # first store the message in message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # track if this is the first message
    is_new_thread = False  

    # Create a new thread only when the first message is sent.
    if st.session_state['thread_id'] is None:
        thread_id = thread_id_generator()
        st.session_state['thread_id'] = thread_id
        title = generate_chat_title(user_input)
        save_thread_title(thread_id, title),
        add_thread(thread_id, title)
        is_new_thread = True

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first store the message in message history
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content 
            for message_chunk, metadata in chatbot.stream(
                {'message': [HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode='messages'
            )
            # generator filter empty content
            if message_chunk.content 
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

    if is_new_thread:
        st.rerun()
