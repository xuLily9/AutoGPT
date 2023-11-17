
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)



# import os
# from apikey import apikey
# import streamlit as st 
# from langchain.llms import OpenAI
# from langchain.prompts import PromptTemplate 
# from langchain.chains import LLMChain, SequentialChain

# os.environ['OPENAI_API_KEY'] = apikey
# #app framework
# st.title('YouTube GPT Creator')
# prompt = st.text_input('Plug in your prompt here')

# # prompt templates 
# title_template = PromptTemplate(
#     input_variables = ['topic'],
#     template ='write me a youtube video title about {topic}'
# )

# script_template = PromptTemplate(
#     input_variables = ['title'],
#     template ='write me a youtube video script based on this title TITLE: {title}'
# )


# llm= OpenAI(temperature = 0.9)
# title_chain = LLMChain(llm=llm, prompt = title_template, verbose = True, output_key='title')
# script_chain = LLMChain(llm=llm, prompt = script_template, verbose = True, output_key='script')
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables={'topic': 'topic'}, output_variables={'title': 'title', 'script': 'script'}, verbose = True)
# # show stuff if there is a prompt, nput_variables =['topic'], output_variables=['title','script'],
# if prompt:
#     response =sequential_chain.run({'topic':prompt})
#     st.write(response['title'])
#     st.write(response['script'])
#     # response =sequential_chain.run({'topic':prompt})
    
