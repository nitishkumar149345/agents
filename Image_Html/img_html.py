from langchain_core.tools import tool
from typing import Annotated
from .openai_vision import get_image_informations
from langchain_openai import OpenAI,ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage,FunctionMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import Graph
from langgraph.prebuilt import ToolInvocation,ToolExecutor
from typing import TypedDict, Annotated, Sequence
# import streamlit as st
from dotenv import load_dotenv
load_dotenv('/Users/omniadmin/Desktop/python-projects/agents/.env')
import os, re
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


llm= ChatOpenAI(model='gpt-3.5-turbo',temperature=0,max_tokens=3500)

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        image : Reference image to generate web page
        prompt : Prompts
        instructions : Description, details of web pagea and instructions to create web page
        initial code : Html code
        final code : final code
    """
    image : str
    code_generation_prompt : str
    code_validation_prompt : str
    instructions : str
    initial_code : str



def instructions_of_webpage(state):
    img_path = state['image']
    # code_generation_prompt = state['code_generation_prompt']
    # code_validation_prompt = state['code_validation_prompt']
    print (img_path)
    instructions= get_image_informations(img_path)
    return {"instructions":instructions}
    # return {"instructions":[instructions],
    #         "code_generation_prompt":code_generation_prompt,
    #         "code_validation_prompt":code_validation_prompt}

def generate_code(state):

    instructions_of_page= state['instructions']
    prompt = ChatPromptTemplate.from_messages([
        ('system',"""
                * Act as an Expert in creating web-pages using HTML and CSS.
                * Your task is to write HTML and CSS code as per the given description.
                * Do not neglect CSS, you has to write both html as well as css in a single file. 
                * If you get any image tags in description, use default url's or file paths.
                * Generate the complete code as a single file having both HTML and CSS.
                * Insure the code perfectly matches with the description.
            """
        ),
        ('user','{input}')
    ])
    
    # base_prompt = state['code_generation_prompt']
    # next_prompt = state['code_validation_prompt']
    # if not base_prompt:
    #     st.warning('no prompt')
    # prompt = ChatPromptTemplate.from_messages([
    #     ('system',prompt),
    #     ('user','{input}')
    # ])
    chain= prompt | llm
    response = chain.invoke({"input":instructions_of_page})

    return {"instructions":instructions_of_page,
            "initial_code":response
           }


def correct_code(state):

    instructions= state['instructions']
    initial_code= state['initial_code']

    prompt= ChatPromptTemplate.from_messages([
        (
            "system","""
            Act as a senior HTML and CSS developer.
            Your task is to verify and correct HTML and CSS code according to the user provided instructions. Return the initial code unchanged if no corrections are needed
            Ensure the initial code meets instructions; if not, make necessary corrections. Also, handle CSS styling as per requirements. 
            Ensure the image tags have been written if the instructions contain images or image tags.and ensure its position, styling, transparency all formatted as per instructions.
            Ensure that the text is in english, if not convert the statements into english.
            Deliver a finalized, complete, and corrected code. 
                """
        ),
        ('user','{input}')
    ])
    # prompt = ChatPromptTemplate.from_messages([
    #     ('system',state['code_validation_prompt']),
    #     ('user','{input}')
    # ])
    chain = prompt | llm
    final_response = chain.invoke({'input':[initial_code,instructions]})
    return final_response
    

workflow= Graph()
workflow.add_node('instructor',instructions_of_webpage)
workflow.add_node('code_generator',generate_code)
workflow.add_node('code_validator', correct_code)

workflow.add_edge('instructor','code_generator')
workflow.add_edge('code_generator','code_validator')


workflow.set_entry_point('instructor')
workflow.set_finish_point('code_validator')


app= workflow.compile()



# inputs = {"image": '/Users/omniadmin/Desktop/python-projects/python/langchain/static/image1.jpeg'}
# result= app.invoke(inputs)
# print ('-'*150)
# print (result)
# code_pattern = r"```html(.*?)```"
# match = re.search(code_pattern, str(result), re.DOTALL)
# # code=''
# if match:
#        code= match.group(1).strip()
#        with open('template.html','w') as page:
#             page.write(code)

# print ('-'*150)
# print (code)