import uuid
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


# ******************************************************* Utility Functions ************************************************************

def thread_id_generator():
    thread_id = uuid.uuid4()
    return thread_id


def new_chat():
    # thread_id = thread_id_generator()
    st.session_state['thread_id'] = None
    # add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_thread_chat(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('message', [])

# ******************************************************* Session State Setup **********************************************************

# st.session_state setup -> dict 
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = None

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# add_thread(st.session_state['thread_id'])

# ********************************************************* Sidebar UI *****************************************************************

st.sidebar.title('LangGraph ChatBot')

if st.sidebar.button('New Chat'):
    new_chat()

st.sidebar.header('My Conversation')

# st.sidebar.text(st.session_state['thread_id'])
for thread_id in reversed(st.session_state['chat_threads']):
    if st.sidebar.button(str(thread_id)):
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

    # Create a new thread only when the first message is sent.
    if st.session_state['thread_id'] is None:
        thread_id = thread_id_generator()
        st.session_state['thread_id'] = thread_id
        add_thread(thread_id)

    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # first store the message in message history
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'message': [HumanMessage(content=user_input)]}, 
                config=CONFIG,
                stream_mode='messages'

            )
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
