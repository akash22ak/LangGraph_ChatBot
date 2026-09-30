import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


# *************************************************st.session_state setup -> dict******************************************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []


# **************************************************Main UI ****************************************************************************

# loading the past chat history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])


user_input = st.chat_input('Type here')
if user_input:

    # first store the message in message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)


    CONFIG = {'configurable': {'thread_id': 'thread-1'}}

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
