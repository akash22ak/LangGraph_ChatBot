import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage


CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# st.session_state -> dict
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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

    response = chatbot.invoke({'message': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_reply = response['message'][-1].content
    # first store the message in message history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_reply})
    with st.chat_message('assistant'):
        st.text(ai_reply)