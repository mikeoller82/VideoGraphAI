from typing import Any, List, Dict
from loguru import logger
import config
import asyncio
from database.db_manager import DatabaseManager
from .agents import Agent, RecentEventsResearchAgent, TitleGenerationAgent, TitleSelectionAgent, DescriptionGenerationAgent, HashtagAndTagGenerationAgent, VideoScriptGenerationAgent, StoryboardGenerationAgent
from .tools import Tool, compile_video_from_storyboard


class Node:
    def __init__(self, agent: Agent = None, tool: Tool = None):
        self.agent = agent
        self.tool = tool
        self.edges: List['Edge'] = []

    async def process(self, input_data: Any) -> Any:
        if self.agent:
            return await self.agent.execute(input_data)
        elif self.tool:
            return await self.tool.use(input_data)
        else:
            raise ValueError("Node has neither agent nor tool")

class Edge:
    def __init__(self, source: Node, target: Node, condition: callable = None):
        self.source = source
        self.target = target
        self.condition = condition

class Graph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        edge.source.edges.append(edge)
        
async def youtube_optimization_workflow(topic: str, duration_minutes: int, time_frame: str, agents: Dict[str, Agent], tools: Dict[str, Tool], db_manager: DatabaseManager) -> Dict[str, Any]:
    graph = Graph()

    # Create nodes
    recent_events_node = Node(agent=agents["RecentEventsResearchAgent"])
    title_gen_node = Node(agent=agents["TitleGenerationAgent"])
    title_select_node = Node(agent=agents["TitleSelectionAgent"])
    desc_gen_node = Node(agent=agents["DescriptionGenerationAgent"])
    hashtag_tag_node = Node(agent=agents["HashtagAndTagGenerationAgent"])
    script_gen_node = Node(agent=agents["VideoScriptGenerationAgent"])
    storyboard_gen_node = Node(agent=agents["StoryboardGenerationAgent"])

    # Add nodes to graph
    for node in [recent_events_node, title_gen_node, title_select_node, desc_gen_node, hashtag_tag_node, script_gen_node, storyboard_gen_node]:
        graph.add_node(node)

    # Create and add edges
    graph.add_edge(Edge(recent_events_node, title_gen_node))
    graph.add_edge(Edge(title_gen_node, title_select_node))
    graph.add_edge(Edge(title_select_node, desc_gen_node))
    graph.add_edge(Edge(desc_gen_node, hashtag_tag_node))
    graph.add_edge(Edge(hashtag_tag_node, script_gen_node))
    graph.add_edge(Edge(script_gen_node, storyboard_gen_node))

    # Execute workflow
    current_node = recent_events_node
    input_data = {"topic": topic, "time_frame": time_frame}
    results = {}
    storyboard = None

    while current_node:
        try:
            logger.info(f"Executing {current_node.agent.name}")
            if isinstance(current_node.agent, VideoScriptGenerationAgent):
                result = await current_node.process((topic, duration_minutes))
            elif isinstance(current_node.agent, RecentEventsResearchAgent):
                result = await current_node.process(input_data)
            else:
                result = await current_node.process(topic if isinstance(current_node.agent, (TitleGenerationAgent, TitleSelectionAgent, DescriptionGenerationAgent, HashtagAndTagGenerationAgent)) else results.get("Video Script Generation Agent", ""))

            logger.info(f"{current_node.agent.name} completed successfully")
            results[current_node.agent.name] = result
            
            # Store result in database
            await db_manager.store_result(current_node.agent.name, result)

            if current_node.agent.name == "StoryboardGenerationAgent":
                storyboard = result if isinstance(result, list) else None

            if current_node.edges:
                current_node = current_node.edges[0].target
                input_data = result
            else:
                break

            await asyncio.sleep(config.get('api_call_delay', 2))

        except Exception as e:
            logger.error(f"Error in {current_node.agent.name}: {str(e)}")
            results[current_node.agent.name] = f"Error: {str(e)}"
            if current_node.edges:
                current_node = current_node.edges[0].target
            else:
                break

            await asyncio.sleep(config.get('error_delay', 5))

    # Always attempt to compile the video, even if there were errors
    try:
        if storyboard is None or not isinstance(storyboard, list) or len(storyboard) == 0:
            logger.warning("No valid storyboard generated. Using placeholder storyboard.")
            storyboard = [{"number": "1", "visual": "Placeholder scene", "text": "Placeholder text", "image_keyword": "placeholder"}]
        
        output_path = compile_video_from_storyboard(storyboard)
        print(f"Storyboard compilation video saved as '{output_path}'")
    except Exception as e:
        logger.error(f"Error compiling video: {str(e)}")
        print(f"Error compiling video: {str(e)}")

    return results
