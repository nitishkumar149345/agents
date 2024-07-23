from Image_Html.img_html import app
from business_planner.main import agent
from Rag_agent.rag import RagAgent

from GPT_RESEARCHER.multi_agents.main import run_research_task
from GPT_RESEARCHER.gpt_researcher.master.actions import stream_output
from GPT_RESEARCHER.gpt_researcher.utils.enum import Tone

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import asyncio
import os, re

api = FastAPI()


origins = ["*"]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



UPLOAD_DIRECTORY = "./uploaded_files"
RAG_DIRECTORY = "./rag_files"
file_paths = []


if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(RAG_DIRECTORY):
    os.makedirs(RAG_DIRECTORY)

@api.get('/img_html', response_class=HTMLResponse)
async def home():
    with open("templates/im_html_home.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


@api.post('/upload')
async def upload_image(file: UploadFile = File()):
    file_path = f'{UPLOAD_DIRECTORY}/{file.filename}'

    with open(file_path,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    inputs = {'image':file_path}

    response = app.invoke(inputs)
    if response and response.content:
        code_pattern = r"`html(.*?)`"
        match = re.search(code_pattern, str(response.content), re.DOTALL)
        if match:
            code = match.group(1).strip()

            return HTMLResponse(content=code)


@api.get('/business_planner', response_class=HTMLResponse)
def business_home():
    with open('templates/business_plan.html', 'r') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


@api.post('/submit')
async def generate_plan(domain: str = Form(...)):
    
    inputs = {'domain':domain}
    result = agent.invoke(inputs)

    reports =[]
    for report in result['final_report']:
        reports.append(report.content)
    return reports


@api.post('/gpt_researcher')
async def gpt_researcher(task: str = Form(...), tone: str = Form(...)):
    tone = Tone[tone]
    report = await run_research_task(query=task,websocket=None, stream_output=stream_output, tone=tone, headers=None)
    print ('*'*100)
    print (report)
    return report['initial_research']



@api.post('/rag')
async def chat_with_docs(files: List[UploadFile] = File(...)):
    global file_paths
    
    for file in files:
        path = os.path.join(RAG_DIRECTORY, file.filename)
        file_paths.append(path)
        with open(path,'wb')as buffer:
            buffer.write(await file.read())
    
@api.post('/chat')
def chat(query: str = Form(...)):
    model = RagAgent(file_paths)
    agent = model.create_chain()
    response = agent.invoke({'input':query})
    return response['answer']





if __name__ == '__main__':
    import uvicorn
#    uvicorn.run(api, host="192.168.0.158",port=8000)
    uvicorn.run(api)
    



