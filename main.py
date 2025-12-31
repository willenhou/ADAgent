import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from adagent.agent import *
from adagent.tools import *
from adagent.utils import *


import os
'''aiskt.pro'''
#os.environ["OPENAI_API_KEY"] = 'sk-3XDOzBFPa484xS2LCb4dEf97351f4308BaDe1985FfEc2418'
#os.environ["OPENAI_BASE_URL"]='https://pro.aiskt.com/v1'

'''aiskt'''
os.environ["OPENAI_API_KEY"] = 'sk-lPUTp51tPBBLWPPd42BbF1780eDc4aCb861140Cd86E71b3d'
os.environ["OPENAI_BASE_URL"]='https://api.zyai.online/v1'

os.environ.get("OPENAI_API_KEY")
os.environ.get("OPENAI_BASE_URL")

warnings.filterwarnings("ignore") # 全局忽略所有的python警告
logging.set_verbosity_error() # 仅显示错误（ERROR）及以上级别的日志，屏蔽 INFO、DEBUG 等低级别日志
_ = load_dotenv() # 从 .env 文件加载环境变量到系统， _ =：忽略返回值（仅利用其副作用）


def initialize_agent(
    prompt_file, tools_to_use=None, model_dir="./model-weights", temp_dir="temp", device="cuda"
):
    """Initialize the ADAgent agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]
    
    '''Current prompt: You are an expert medical AI assistant. 
        Make multiple tool calls in parallel or sequence as needed for comprehensive answers.
        Critically think about and criticize the tool outputs.
        If you need to look up some information before asking a follow up question, you are allowed to do that!'''

    all_tools = {
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        "MriDiagnosisTool": lambda: MriDiagnosisTool(),
        "PetDiagnosisTool": lambda: PetDiagnosisTool(),
        "MriPetDiagnosisTool": lambda: MriPetDiagnosisTool()

    }

    # Initialize only selected tools or all if none specified
    tools_dict = {}
    tools_to_use = tools_to_use or all_tools.keys() #If tools_to_use is "truthy", then tools_to_use retains its current value.
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    checkpointer = MemorySaver()
    model = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the ADAgent application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool",
        "DicomProcessorTool",
        # "ChestXRayClassifierTool",
        # "ChestXRaySegmentationTool",
        # "ChestXRayReportGeneratorTool",
        #"XRayVQATool",
        "MriDiagnosisTool",
        "PetDiagnosisTool",
        "MriPetDiagnosisTool"
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",
    ]

    from pathlib import Path
    project_root = Path(__file__).parent.resolve()
    agent, tools_dict = initialize_agent(
        str(project_root / "adagent" / "docs" / "system_prompts.txt"), 
        tools_to_use=selected_tools, 
        model_dir=str(project_root / "model-weights")
    )
    #print(tools_dict)
    #print(tools_dict.values())
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=None, share=True) #8585
