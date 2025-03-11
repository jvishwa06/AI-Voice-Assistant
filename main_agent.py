import os
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

# Load environment variables
_ = load_dotenv(find_dotenv())

# Configuration settings
defaults = {
    "groq_api_key": os.getenv("GROQ_API_KEY"),
    "serpapi_api_key": os.getenv("SERPAPI_API_KEY"),
    "model": "llama-3.1-8b-instant",
    "temperature": 0.7,
    "voice": "com.apple.eloquence.en-US.Sandy",
    "volume": 1.0,
    "rate": 200,
    "session_id": "abc123",
    "ability": "general knowledge",
}

# Validate API keys
if not defaults["groq_api_key"]:
    raise ValueError("You must set the GROQ_API_KEY environment variable to use the Groq API")
if not defaults["serpapi_api_key"]:
    raise ValueError("You must set the SERPAPI_API_KEY environment variable to use SerpAPI")

# Initialize the Groq Chat API client
llm = ChatGroq(
    temperature=defaults["temperature"],
    model=defaults["model"],
    api_key=defaults["groq_api_key"]
)

search = GoogleSerperAPIWrapper(serper_api_key=defaults["serpapi_api_key"])

# Define tools for the agent
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Use this tool to search the web for up-to-date information."
    )
]

# Initialize the agent with tools, set return_intermediate_steps to True
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True
)

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You're an assistant who's good at {ability}. Respond in 30 words or fewer."),
        ("ai", "Hello, I am Sivanirai2.0. How can I assist you today?"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

# Initialize the runnable chain with memory management
runnable = prompt | agent
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', defaults["voice"])
engine.setProperty('volume', defaults["volume"])
engine.setProperty('rate', defaults["rate"])

# Initialize speech recognition engine
recognizer = sr.Recognizer()

def speak(text):
    print("Sivanirai2.0: " + text)
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
        config={"configurable": {"session_id": defaults["session_id"]}}
    )

    # Extract the final answer from the response
    if isinstance(response, dict):
        content = response.get('output', 'No output found')
    else:
        content = getattr(response, 'content', 'No content found')

    return content

# Greet the user
speak("Hello, I am Sivanirai2.0. How can I assist you today?")

# Main interaction loop
while True:
    user_input = listen()
    if user_input:
        if user_input.lower() in ["exit", "quit", "thank you for your help"]:
            speak("You're welcome! Have a great day!")
            break
        response = generate_response(defaults["ability"], user_input)
        for sentence in response.split("."):
            speak(sentence.strip())
    else:
        speak("I'm sorry, I didn't understand that.")