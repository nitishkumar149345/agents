import base64
from langchain.chains import TransformChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAI, ChatOpenAI
from langchain import globals

from dotenv import load_dotenv
load_dotenv('/Users/omniadmin/Desktop/python-projects/agents/.env')
import os
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

class ImageInformation(BaseModel):
    """
    Think the image you got as a Web-Page.
    Information of the web-page(image) related to HTML and CSS"""

    Page_description: str = Field(description=""" A detailed description of the web page, focusing on the visual and structural aspects as they relate to HTML and CSS. 
                                   This includes the overall layout, thematic elements, and specific sections of the page.""")

    Elements: list[str] = Field(description=""" A comprehensive list of the various elements and sections on the web page. This includes layout structures (such as header, navigation menu, hero section), 
                                specific HTML elements (such as div blocks, images, buttons, search), CSS styling details (such as colors, fonts, text alignment, transparency), and responsive design considerations.""")
    
    Text: list[str] = Field(description="""A list of all the text present on the web page.Make the text only in english. Each entry should detail the exact text content, its line by line position
                            its HTML tag (e.g., h1, h2, p), font size,font style, font weight, color, and any other relevant CSS styling (e.g., uppercase, centered).""")

    Images: list[str] = Field(description=""" Details of the images, background present in the webpage. It should contain all the info related to images like, its position(top, left corner, right cornet,etc),
                              styling, transparency,overlay,shape, consider size of the image by how much space an it occupied like half, quter the page. """)
    
    Instructions: list[str] = Field(description="""Step-by-step instructions to recreate the web page using HTML and CSS. This should be a detailed guide covering:
                                * HTML structure: defining the elements and their hierarchy.
                                * CSS styling: specifying the styles for each element, including layout, colors, fonts, and responsiveness.
                                * Layout considerations: ensuring the page looks good on both desktop and mobile devices.
                                * Colors of each text tag,elements, sections, overall page.""")

    CSS_Styling: str = Field(description=""" The css styling of the webpage. Including 
                             - Layout and Positioning: flexbox/grid, positioning, margins and padding
                             - Typography: Fonts, Text Alignment 
                             - Color and Background: color, backgroung-color, background-image,background-postion
                             - Borders and Shadows: Borders, box shadows
                             - Box Model: content-box,border-box,Content, Padding, Border, Margin:""")                

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""
    image_path = inputs["image_path"]
    image_base64 = encode_image(image_path)
    return {"image": image_base64}



load_image_chain = TransformChain(
    input_variables = ['image_path'],
    output_variables = ['image'],
    transform = load_image
) 

globals.set_debug(True)

@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1024)
    parser = JsonOutputParser(pydantic_object=ImageInformation)

    msg = model.invoke(
                [HumanMessage(
                content=[
                {"type": "text", "text": inputs["prompt"]},
                {"type": "text", "text": parser.get_format_instructions()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
                ])]
                )
    return msg.content

parser = JsonOutputParser(pydantic_object=ImageInformation)
def get_image_informations(image_path: str, ) -> dict:
   vision_prompt = """
   Think the provided image as a web page, provide this information:
   - page description
   - Html elements
   - Images
   - Text 
   - CSS Styling
   
   """
   print ('generating instructions.........................')
   vision_chain = load_image_chain | image_model | parser
   return vision_chain.invoke({'image_path': f'{image_path}', 
                               'prompt': vision_prompt})

# result = get_image_informations("/Users/omniadmin/Desktop/python-projects/python/langchain/static/image2.jpeg")
# print ('-'* 150)
# print (result)