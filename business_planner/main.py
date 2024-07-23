from .marketing_agent import MarketingAgent
from .business_agent import BusineesAgent
from .technical_agent import TechnicalAgent

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
load_dotenv('.env')
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ['TAVILY_API_KEY'] = os.getenv('TAVILY_API_KEY')


class AgentState(TypedDict):
    """
    Reprasents the state of our graph.

    Attributes:
        domain: Domain in which user wants to start business
        business_model: Best business model to start with the domain
        final_report: Reports generated by the agents
        """
                    
                    
    domain : str
    business_model : str
    final_report :  Annotated[list, add_messages]


reports=[]

def marketing_agent(state:AgentState):
    domain = state['domain']
    agent= MarketingAgent(domain)
    response= agent()
    # reports.append(response['marketing_report'])
    
    return {'business_model':response['business_model'], 'final_report':response['marketing_report']}

def business_agent(state:AgentState):
    business_model = state['business_model']
    agent = BusineesAgent(business_model)
    response = agent()
    # reports.append(response)
    return {'business_model':business_model,'final_report':response}

def technical_agent(state:AgentState):
    business_model = state['business_model']
    agent = TechnicalAgent(business_model)
    response = agent()
    
    return {'final_report':response}


   
workflow = StateGraph(AgentState)
workflow.add_node('marketing_agent', marketing_agent)
workflow.add_node('business_agent', business_agent)
workflow.add_node('technical_agent', technical_agent)


workflow.add_edge('marketing_agent','business_agent')
workflow.add_edge('business_agent','technical_agent')


workflow.set_entry_point('marketing_agent')
workflow.set_finish_point('technical_agent')

agent = workflow.compile()




# inputs = {'domain':'Large Language Modles'}
# result = agent.invoke(inputs)
# print ('-'*100)
# print (reports)
