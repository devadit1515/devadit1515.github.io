import streamlit as st
import openai
from config import OPENAI_API_KEY
import json

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def initialize_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Mango Farm Assistant. How can I help you today?"}
        ]

def display_chat():
    # Chat container
    with st.container():
        st.markdown("### 🤖 Mango Farm Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about mango farming..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Call OpenAI API
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful assistant for mango farmers. Provide concise, practical advice about mango cultivation, weather impact, and farm management."}
                            ] + [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages
                            ]
                        )
                        
                        # Get the assistant's response
                        assistant_response = response.choices[0].message.content
                        
                        # Display and store the assistant's response
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
