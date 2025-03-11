import os
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configuration settings
defaults = {
    "api_key": os.getenv("GROQ_API_KEY"),
    "model": "llama-3.1-8b-instant",
    "temperature": 0.7,
    "voice": "com.apple.eloquence.en-US.Sandy",
    "volume": 1.0,
    "rate": 200,
    "session_id": "abc123",
    "ability": "Psychology",
    "base_url": "https://api.groq.com/v1",
}

# Extract configuration values
temperature = defaults["temperature"]
interface_voice = defaults["voice"]
volume = defaults["volume"]
rate = defaults["rate"]
session_id = defaults["session_id"]
ability = defaults["ability"]
api_key = defaults["api_key"]

# Validate API key
if not api_key:
    raise ValueError("You must set the GROQ_API_KEY environment variable to use the Groq API")

# Initialize the Groq Chat API client
llm = ChatGroq(temperature=temperature, model=defaults["model"], api_key=api_key)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}. Respond in 30 words or fewer"),
        ("ai", "Hello, I am SANDY. How can I help you today?"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

# Initialize the runnable chain with memory management
runnable = prompt | llm
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', interface_voice)
engine.setProperty('volume', volume)
engine.setProperty('rate', rate)

# Initialize speech recognition engine
recognizer = sr.Recognizer()

def speak(text):
    print("SANDY: " + text)
    engine.say(text)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio)
        print(f"You: {text}")
        return text
    except sr.UnknownValueError:
        print("I'm sorry, I didn't catch that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def generate_response(ability, user_input):
    response = with_message_history.invoke(
        {"ability": ability, "input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content

speak("Hello, I am SANDY. How can I help you today?")

while True:
    print("Listening...")
    prompt = listen()
    if prompt:
        print("You: " + prompt)
        if prompt.lower() == "thank you for your help":
            exit()
        response = generate_response(ability, prompt)
        sentences = response.split(".")
        for sentence in sentences:
            speak(sentence.strip())
    else:
        speak("I'm sorry, I didn't understand that.")
