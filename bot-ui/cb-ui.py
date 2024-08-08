#!/usr/bin/python3

# Import necessary libraries
import streamlit as st
from PIL import Image
import requests, json, random, logging, time, datetime

# Constants ============================================
BASE_URL="http://localhost:9000"
NOAUTH_HEADERS = { "Content-Type": "application/json" }
PAGE_TITLE = "🤖 ConsultaBot! (BLUE) 🧠 "
THINKING_MSGS = [
    "Gimme a sec to research that...",
    "Mulling this over, uno momento..",
    "Let me gather my thoughts on that...",
    "Thanks for your input, pondering...",
    "Please allow me a moment to compose my thoughts...",
]
LOGFILE = "./logs/cb-ui.log"
FEEDBACKLOG = "./logs/feedback.log"

# ============================================

loglevel = logging.INFO
logfmode = 'w'                # w = overwrite, a = append
logging.basicConfig(filename=LOGFILE, encoding='utf-8', level=loglevel, filemode=logfmode)

# ============================================
# Setup Streamlit page configuration
im = Image.open('sricon.png')
st.set_page_config(page_title=PAGE_TITLE, layout='wide', page_icon = im)
# Initialize session states
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []
if "just_sent" not in st.session_state:
    st.session_state["just_sent"] = False
if "temp" not in st.session_state:
    st.session_state["temp"] = ""

# Set up the Streamlit app layout
st.title(PAGE_TITLE)
#st.subheader(" Powered by 🦜 LangChain + LLama + Streamlit")

hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

# Add a button to start a new chat
#st.sidebar.button("New Chat", on_click = new_chat, type='primary')

#####################################################
def clear_text():
    st.session_state["temp"] = st.session_state["input"]
    st.session_state["input"] = ""

#####################################################
# Define function to get user input
def get_text():
    input_text = st.text_input("You: ",
                            st.session_state["input"],
                            key="input", 
                            placeholder="ConsultaBot is here to help! Ask me anything related to Conjur Cloud...", 
                            on_change=clear_text,    
                            label_visibility='hidden')
    input_text = st.session_state["temp"]
    return input_text

#####################################################
# Clears session state and starts a new chat.
def new_chat():
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state.entity_memory.store = {}
    st.session_state.entity_memory.buffer.clear()

user_text = get_text()
if user_text:
    logging.info(f"user_text: {user_text}")
    with st.spinner(random.choice(THINKING_MSGS)):
        user_input = {"data": user_text}
        url = BASE_URL+"/query"
        payload = json.dumps(user_input)
        llm_output = json.loads(requests.request("POST", url, headers=NOAUTH_HEADERS, data=payload).text)
    logging.info(f"llm_output: {llm_output}")
    st.session_state.past.append(user_text)  
    st.session_state.generated.append(llm_output)

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Chat History:", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i],icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")
                            
# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

with st.sidebar:
    fbform = st.form('feedback_form', clear_on_submit=False)
    with fbform:
        st.markdown("# Feedback")
        uname = st.text_input("Your name/email:")
        feedback = st.text_area('How can we improve ConsultaBot?')
        if st.form_submit_button('Submit feedback'):
            fblog = open(FEEDBACKLOG, "a")
            fblog.write("\n\n" + "-"*40 + "\n")
            fblog.write(datetime.datetime.now().strftime("%I:%M%p %B %d, %Y") + "\n")
            fblog.write("Name: " + uname + "\n")
            fblog.write(feedback + "\n")
            fblog.close()
            fbform.subheader("Feedback logged. Thanks!")
            time.sleep(2)
            fbform.empty()

    st.markdown("# Caveats & Disavowals")
    st.markdown(
       "This tool is very much a work in progress. It has not undergone prompt engineering or other fine tuning. It is entirely possible it will provide false, incorrect or misleading information. CyberArk is not responsible for any actions taken based on responses generated by this software."
    )

    st.markdown("---")
    st.markdown("# About")
    st.markdown('''
    ConsultaBot is a RAG LLM chatbot, self-hosted on a single Ubuntu 22.04 VM. 
    It is blocked from making any off-host API calls.
    Its knowledgebase is currently limited to Conjur Cloud docs and blog articles.
    <br><br>
    It will only respond to input it perceives as relevant to its knowledgebase.
    '''
    , unsafe_allow_html=True
    )

    st.markdown("# Change History")
    change_log='''
    24-08-08:<br>
      - fixed empty response bug (?)<br>
    24-08-05:<br>
      - KB docs in separate repo<br>
      - Added Lllama 3 system prompt<br>
      - refactored retrieval chain<br>
      - Maxxed out max_tokens<br>
    24-08-02:<br>
      - Validate query relevance<br>
      - Thinking spinner<br>
      - B/G deployments<br>
      - feedback form/log<br>
    '''
    cl_table = f"<span style='font-size:12px;'>{change_log}</span>"
    st.markdown(cl_table, unsafe_allow_html=True)