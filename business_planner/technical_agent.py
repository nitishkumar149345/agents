from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('./env')
from langchain_core.output_parsers import JsonOutputParser
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


# os.environ['LANGCHAIN_ENDPOINT']='https://api.smith.langchain.com'
# os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["LANGCHAIN_TRACING_V2"] = 'true'
# os.environ["LANGCHAIN_PROJECT"] = 'default'

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

class Queries(BaseModel):
    questions : list[str] = Field(description='A list of detailed and relevant questions that can help gather comprehensive information about a business model')



class TechnicalAgent():
    def __init__(self, business_model) :
        self.business_model = business_model
        self.observations = 'Technical Report: '


    def generate_questions(self):

        parser = JsonOutputParser(pydantic_object=Queries)

        prompt = ChatPromptTemplate.from_messages([
            ('system','''
             
                Your task is to generate a list of detailed and relevant questions that will help gather comprehensive information about the technical needs of a specified business model.
                The questions should cover the following aspects to ensure a complete technical report:

                - Technical Solutions and Recommendations :- Necessary skills and technologies needed to implement the business. Practical solutions for the development team.
                - Sample Python Code snippets :- Relevant python code snippets and examples needed to implement the business.
                - Technology Recommendations :- Suggested technologies and tools for implementation.
                - Techincal roadmap

                You will receive the business model, and based on that, you need to generate the questions as mentioned above.
                \nformat_instructions: {format_instructions}
                '''),
            ('user','{model}')
        ])

        chain = prompt | llm | parser
        result = chain.invoke({'model':self.business_model, 'format_instructions':parser.get_format_instructions()})
        self.questions = result['questions']

    def agent(self):
        prompt = ChatPromptTemplate.from_messages([

            ('system','''
                Role :- Technical Developer
                Goal :- Generate informative and comprehensive answers for the user questions based on your research
                Task :- Do the research to acquire the related information to answer the user question.
                        The answer should be descriptive, comprehensive and informative. The answers should be detailed.
                        Avoid answering with straight and point answers.
                Tools:- You have access to this tool 'DuckDuckGoSearchRun' to do online searching.
                    '''),

            ('user','{question}'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        tools = [DuckDuckGoSearchRun()]
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
  

        for question in self.questions:
            response = agent_executor.invoke({'question':question})
            self.observations+=response['output']
       
        
    
    def __call__(self):
        self.generate_questions()
        self.agent()
        print ('---------------formatting report-----------------')
        prompt = ChatPromptTemplate.from_messages([
            ('system','''

                Role :- Report Writer
                Goal :- Your sole purpose is to write a well-written technical reports about a business model based on research findings and information
                Task :- You are tasked with formatting the provided input into a technical report.
                        The report must adhere strictly to the existing content without adding, removing, or summarizing any information.
                        Ensure all sections are properly titled and organized, and format any Python code snippets appropriatelys
                output:- The output must be well formatted and well written technical report, so that report can be directly used in docs.
                warning :- Do not add, remove, or summarize any content. Restructure the input only to organize it with clear titles and headings as specified.
                '''),

            ('user','{input}')
        ])
    
        chain = prompt | llm
        response = chain.invoke({'input':self.observations})
        return response.content

# agent = TechnicalAgent('Subscription-Based Model for Large Language Models')
# re = agent()
# print (re)