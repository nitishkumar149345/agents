from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
load_dotenv('/Users/omniadmin/Desktop/python-projects/agents/.env')
from langchain_core.output_parsers import JsonOutputParser
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

class Queries(BaseModel):
    questions : list[str] = Field(description='A list of detailed and relevant questions that can help gather comprehensive information about a business model')

class BusinessModel(BaseModel):
    business_model : str = Field(description='Best business model on the provided domain')
    description : str = Field(description='description about the business model')


class MarketingAgent():
    def __init__(self, domain) :
        self.domain = domain
        self.observations = 'Marketing Report: '

    def find_business_model(self):
        
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                Role :- Marketing Expert
                Goal :- Find the best business model based on the provided domain
                Task :- Analyze and identify the best business model for the provided domain.
                        This task involves researching current market trends, competitor analysis,
                        and potential opportunities within the domain.
                output :- Return only the identifyed business model. Should not retrun any other thing. just name of the business model you suggest.
            '''),
            ('user','{domain}'),
            MessagesPlaceholder(variable_name='agent_scratchpad')
        ])

        tools = [DuckDuckGoSearchRun()]
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        response = agent_executor.invoke({'domain':self.domain})
        self.business_model = response['output']
        

    def generate_questions(self):

        parser = JsonOutputParser(pydantic_object=Queries)

        prompt = ChatPromptTemplate.from_messages([
            ('system','''

                Generate a list of detailed and relevant questions that can help gather comprehensive information about a business model.
                The questions should cover various aspects such as the Current Market Trends, Competitor Analysis,Revenue Streams,Target Customer Segments,
                Value Propositions, Potential Risks and Opportunities and other business model related.
                You will recieve the business model, then you have to generate the questions as mentioned.
                \nformat_instructions: {format_instructions}
                '''),
            ('user','{domain}')
        ])

        chain = prompt | llm | parser
        result = chain.invoke({'domain':self.business_model, 'format_instructions':parser.get_format_instructions()})
        self.questions = result['questions']

    def agent(self):
        prompt = ChatPromptTemplate.from_messages([

            ('system',''' 
                Role :- Marketing Researcher
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

        self.observations = 'business mdoel---'+self.business_model

        for question in self.questions:
            response = agent_executor.invoke({'question':question})
            self.observations+=response['output']

        
    
    def __call__(self):
        self.find_business_model()
        self.generate_questions()
        self.agent()
        # print ('---------------formatting report-----------------')
        prompt = ChatPromptTemplate.from_messages([
            ('system','''
                    Role :- Content Writer
                    Goal :- Format the given input into a nice marketing report
                    Task :- You are tasked with formatting the provided input into a marketing report.
                            The report must adhere strictly to the existing content without adding, removing, or summarizing any information.
                            Ensure all sections are properly titled and organized, and format any Python code snippets appropriatelys.
                    output:- The output must be well formatted and well written marketing report, so that report can be directly used in docs.
            
                    warning :- Do not add, remove, or summarize any content. Restructure the input only to organize it with clear titles and headings as specified.
                    '''),
            ('user','{input}')
        ])

        chain = prompt | llm
        response = chain.invoke({'input':self.observations})
        return {'business_model':self.business_model, 'marketing_report':response.content}

# agent = MarketingAgent('large language models with langchain and langgraph')

# re = agent()
# print (re)
        
