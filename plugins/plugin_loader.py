import importlib
import os
from .agents import Agent
from .tools import Tool
    

def load_plugins():
    agents = {}
    tools = {}
    
    for filename in os.listdir('plugins'):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]
            module = importlib.import_module(f'plugins.{module_name}')
            
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type):
                    if issubclass(item, Agent) and item != Agent:
                        agents[item_name] = item
                    elif issubclass(item, Tool) and item != Tool:
                        tools[item_name] = item
    
    return agents, tools
