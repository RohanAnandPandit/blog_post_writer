from blog_agents.blog_generator import blog_generator_agent
import streamlit as st
import os
from blog_agents.blogpostcreator import BlogPostCreator
from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner
import asyncio

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

with st.sidebar:
    "## ✍️ Blog Post Generator"

    
    os.environ['OPENAI_API_KEY'] = st.secrets.openai.api_key

    st.divider()

    """
    ### About

    ✍️ Blog Post Generator allows you to generate an SEO optimised blog post for your website. 
    """

    st.divider()

# Initialize session state for chat history, blog topic, response, and recommended topics
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to add messages to chat history
def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

# Function to generate response (non-async wrapper)
def generate_response(messages):
    # Use the existing event loop instead of creating a new one
    result = loop.run_until_complete(Runner.run(blog_generator_agent, input=messages))
    return result.final_output

# Initial message from the chatbot
if not st.session_state.messages:
    add_message("assistant", "Hello! Please enter your website URL to recommend blog topics.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message here..."):
    add_message("user", prompt)

    with st.chat_message('user'):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            output = generate_response(st.session_state.messages)
            st.markdown(output)
            add_message("assistant", output)

    
