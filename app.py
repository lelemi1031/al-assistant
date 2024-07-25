from dotenv import load_dotenv
import streamlit as st
# from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import os
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template


from utils.utils import init_llm
from utils.websearch import get_embedding, get_conversation_chain_v1, get_conversation_chain_v2, search

LANG = "EN" 


# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")


# # Function to process user input and generate response using LlamaCpp
# def process_input(input_text, prompt_template, llama_cpp_instance, dialogue, embedding):
#     # Process input using LlamaCpp
#     dialogue += f"*Q* {input_text}\n"
#     prompt = prompt_template.format(dialogue=dialogue)
#     if input_text.startswith("search"):
#         st.write("searching")
#         response = get_conversation_chain(input_text, llama_cpp_instance, embedding)
#         reply = response["answer"]
#         print(reply)
#     else:
#         reply = llama_cpp_instance.invoke(prompt, max_tokens=512)

#     if reply is not None:
#         dialogue += f"*A* {reply}\n"
#     return reply, dialogue
    

def handle_userinput(user_question, use_microphone):

    response = st.session_state.conversation({'question': user_question})
    print(response)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            if i == len(st.session_state.chat_history) - 1:
                resource = '\n'.join([i.metadata['source'] for i in response['source_documents']])
                answer = f" {message.content} \n \n  References: \n {resource}"
            else:
                answer = message.content

            st.write(bot_template.replace(
                "{{MSG}}", answer), unsafe_allow_html=True)
            # Convert response to speech and play
            # st.write("Generating...")
            if use_microphone:
                text_to_speech(message.content)
            
            

# Function to handle text input
def handle_text_input():
    submitted_text = st.session_state.input_text
    
    handle_userinput(submitted_text, use_microphone=False)
    # st.write("You submitted:", submitted_text)
    # Clear the input text
    st.session_state.input_text = ""


# Streamlit app
def main_page():
    load_dotenv()

    st.set_page_config(page_title="Voice Assistant Chat App",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # llama_cpp_instance = init_llm()  # Initialize LlamaCpp instance
    llama_cpp_instance = None
    embedding = get_embedding()  # Get embeddings

    # Add a system prompt
    # dialogue = "You are a helpful assistant named Linda. You are here to help the user with any questions they have."

    with st.sidebar:
        st.header("AI Assisstant Linda")
        use_microphone = st.checkbox('use microphone')
        voice_input = None
        
        if not use_microphone:
            user_input = st.text_input("Ask anything!", key='input_text', on_change=handle_text_input)
 
    
    if use_microphone:
        while True:
            # Record user's voice
            r = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Speak something...")
                audio = r.listen(source)

            # Convert speech to text
            try:
                voice_input = r.recognize_whisper(audio, language="english")
                
            except sr.UnknownValueError:
                st.write("Sorry, could not understand audio.")
                pass

            try:
                if voice_input:
                    handle_userinput(voice_input, use_microphone=True)
            except Exception as e:
                st.write("Error in processing user input: ", e)
                pass


    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        st.session_state.conversation = get_conversation_chain_v1(
         embedding, llama_cpp_instance)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if 'input_text' not in st.session_state:
        st.session_state['input_text'] = ""


if __name__ == "__main__":
    main_page()
