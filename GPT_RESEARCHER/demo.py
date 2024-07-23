from fastapi import FastAPI
import datetime

from multi_agents.main import run_research_task
from gpt_researcher.master.actions import stream_output
import asyncio
from dotenv import load_dotenv
load_dotenv('/Users/omniadmin/Desktop/python-projects/agents/.env')
from gpt_researcher.utils.enum import ReportType, Tone
import os

os.environ['TAVILY_API_KEY'] = "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"
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')




async def run_agent(task, report_type, tone:Tone, websocket, headers=None):
    start_time = datetime.datetime.now()
   
    tone = Tone[tone]
    if report_type == "multi_agents":
        report = await run_research_task(query=task, websocket=websocket, stream_output=stream_output, tone=tone, headers=headers)
        

    return report


r = asyncio.run( run_agent('best stocks to buy', 'multi_agents','Objective',None,None ))
print ('*'*100)
print (r['initial_research'])