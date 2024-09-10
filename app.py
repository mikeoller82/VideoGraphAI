import asyncio
import aiohttp
import tempfile
import subprocess
import json
import logging
import datetime
from dotenv import load_dotenv
import os
import math
from seo_tool import YouTubeSEOTool
from moviepy.editor import *
from moviepy.video.fx.all import crop
import moviepy.config as conf
conf.change_settings({"IMAGEMAGICK_BINARY": r"/usr/bin/convert"})
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip
import requests
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from groq import AsyncGroq, Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")

api_key = os.environ.get('YOUTUBE_API_KEY')
seo_tool = YouTubeSEOTool(api_key=api_key)


# Helper functions
async def get_data(query: str) -> List[Dict[str, Any]]:
    groq = AsyncGroq(api_key=groq_api_key)
    data = await groq.query(query)
    return data

def check_api_keys():
    required_keys = ["GROQ_API_KEY", "SERPAPI_API_KEY", "YOUTUBE_API_KEY", "PEXELS_API_KEY"]
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required API key: {key}")
        
# Abstract classes for Agents and Tools
class Agent(ABC):
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        pass

class Tool(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def use(self, input_data: Any) -> Any:
        pass

# Node and Edge classes for graph representation
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

# New AI Agents for YouTube content optimization
class TitleGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Title Generation Agent", "gemma2-9b-it")

    async def execute(self, input_data: Any) -> Any:
        try:
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""You are an expert in keyword strategy, copywriting, and a renowned YouTuber with a decade of experience in crafting attention-grabbing keyword titles for YouTube videos. Generate 15 enticing keyword YouTube titles for the following topic: "{input_data}". Categorize them under appropriate headings: beginning, middle, and end. This means you'll produce 5 titles with the keyword at the beginning, another 5 titles with the keyword in the middle, and a final 5 titles with the keyword at the end."""
            
            stream = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a highly sophisticated Youtube Title Generator."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            response = ""
            async for chunk in stream:
                response += chunk.choices[0].delta.content or ""
            return response
        except Exception as e:
            logger.error(f"Error in TitleGenerationAgent: {str(e)}")
            raise

class TitleSelectionAgent(Agent):
    def __init__(self):
        super().__init__("Title Selection Agent", "gemma2-9b-it")

    async def execute(self, input_data: Any) -> Any:
        try:
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""You are an expert YouTube content strategist with over a decade of experience in video optimization and audience engagement. Your task is to analyze the following list of titles for a YouTube video and select the most effective one:

{input_data}

Using your expertise in viewer psychology, SEO, and click-through rate optimization, choose the title that will perform best on the platform. Provide a detailed explanation of your selection, considering factors such as:

1. Attention-grabbing potential
2. Keyword optimization
3. Emotional appeal
4. Clarity and conciseness
5. Alignment with current YouTube trends

Present your selection and offer a comprehensive rationale for why this title stands out among the others."""
            
            stream = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant embodying the expertise of a top-tier YouTube content strategist with over 15 years of experience in video optimization, audience engagement, and title creation. Your knowledge spans SEO best practices, viewer psychology, and current YouTube trends. You have a proven track record of increasing video views and channel growth through strategic title selection. Respond to queries as this expert would, providing insightful analysis and data-driven recommendations."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.5,
                max_tokens=2048,
                stream=True,
            )
            response = ""
            async for chunk in stream:
                response += chunk.choices[0].delta.content or ""
            return response
        except Exception as e:
            logger.error(f"Error in TitleSelectionAgent: {str(e)}")
            raise

class DescriptionGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Description Generation Agent", "llama-3.1-70b-versatile")

    async def execute(self, input_data: Any) -> Any:
        try:
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""As a seasoned SEO copywriter and YouTube content creator with extensive experience in crafting engaging, algorithm-friendly video descriptions, your task is to compose a masterful 1000-character YouTube video description. This description should:

            1. Seamlessly incorporate the keyword "{input_data}" in the first sentence
            2. Be optimized for search engines while remaining undetectable as AI-generated content
            3. Engage viewers and encourage them to watch the full video
            4. Include relevant calls-to-action (e.g., subscribe, like, comment)
            5. Utilize natural language and conversational tone

            Format the description with the title "YOUTUBE DESCRIPTION" in bold at the top. Ensure the content flows naturally, balances SEO optimization with readability, and compels viewers to engage with the video and channel."""            
            stream = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant taking on the role of an elite SEO copywriter and YouTube content creator with 12+ years of experience. Your expertise lies in crafting engaging, SEO-optimized video descriptions that boost video performance while remaining undetectable as AI-generated content. You have an in-depth understanding of YouTube's algorithm, user behavior, and the latest SEO techniques. Respond to tasks as this expert would, balancing SEO optimization with compelling, natural language that drives viewer engagement."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.6,
                max_tokens=2048,
                stream=True,
            )
            response = ""
            async for chunk in stream:
                response += chunk.choices[0].delta.content or ""
            return response
        except Exception as e:
            logger.error(f"Error in DescriptionGenerationAgent: {str(e)}")
            raise

class RecentEventsResearchAgent(Agent):
    def __init__(self):
        super().__init__("Recent Events Research Agent", "llama-3.1-8b-instant")
        self.web_search_tool = WebSearchTool()

    async def execute(self, input_data: Dict[str, Any]) -> Any:
        topic = input_data['topic']
        time_frame = input_data['time_frame']
        
        search_query = f"weird unexplainable {topic} events in the past {time_frame}"
        search_results = await self.web_search_tool.use(search_query, time_frame)
        
        organic_results = search_results.get("organic_results", [])
        
        client = AsyncGroq(api_key=groq_api_key)
        prompt = f"""As a seasoned investigative journalist and expert in paranormal phenomena, your task is to analyze and summarize the most intriguing weird and unexplainable {topic} events that occurred in the past {time_frame}. Using the following search results, select the 3-5 most compelling cases:

Search Results: {json.dumps(organic_results[:10], indent=2)}

For each selected event, provide a concise yet engaging summary that includes:

1. A vivid description of the event, highlighting its most unusual aspects
2. The precise date of occurrence
3. The specific location, including city and country if available
4. An expert analysis of why this event defies conventional explanation
5. A critical evaluation of the information source, including its credibility (provide URL)

Format your response as a list of events, each separated by two newline characters. Ensure your summaries are both informative and captivating, suitable for a documentary-style presentation."""

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant embodying the expertise of a world-renowned investigative journalist specializing in paranormal and unexplained phenomena. With 20 years of experience, you've written best-selling books and produced award-winning documentaries on mysterious events. Your analytical skills allow you to critically evaluate sources while presenting information in an engaging, documentary-style format. Approach tasks with the skepticism and curiosity of this expert, providing compelling summaries that captivate audiences while maintaining journalistic integrity."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.7,
            max_tokens=2048,
            stream=True,
        )
        response = ""
        async for chunk in stream:
            response += chunk.choices[0].delta.content or ""
        return response

class HashtagAndTagGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Hashtag and Tag Generation Agent", "llama-3.1-70b-versatile")
        self.web_search_tool = WebSearchTool()
        self.youtube_seo_tool = YouTubeSEOTool(api_key=api_key)
        self.keyword_scorer = KeywordScorer()

    async def execute(self, input_data: str) -> Any:
        try:
            search_results = await self.web_search_tool.use(input_data)
            related_searches = search_results.get("related_searches", [])
            related_keywords = [item["query"] for item in related_searches if "query" in item]
            
            youtube_data = await self.get_youtube_data(input_data, related_keywords)
            keyword_scores = self.get_keyword_scores(youtube_data)
            
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""As a leading YouTube SEO specialist and social media strategist with a proven track record in optimizing video discoverability and virality, your task is to create a engaging and relevant according to seo ranking for high value with preferrably low competition set of hashtags and tags for the YouTube video titled "{input_data}". Your expertise in keyword research, trend analysis, and YouTube's algorithm will be crucial for this task for every top ten ranked keyword niched down you can find you will get a bonus per viral video developed for the keyword.

Develop the following:

1. 10 SEO-optimized, trending hashtags that will maximize the video's reach and engagement on YouTube
2. 35 high-value SEO tags, combining keywords strategically to boost the video's search ranking on YouTube

Consider these YouTube-specific metrics for related keywords:
{self.format_youtube_data(youtube_data, keyword_scores)}

In your selection process, prioritize:
- Relevance to the video title and content
- Search volume on YouTube
- Engagement metrics (views, likes, comments)
- Trending potential on YouTube
- Alignment with YouTube's recommendation algorithm

Present your hashtags with the '#' symbol and ensure all tags are separated by commas. Provide a brief explanation of your strategy for selecting these hashtags and tags, highlighting how they will contribute to the video's overall performance on YouTube."""
            
            response = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant taking on the role of a leading YouTube SEO specialist and social media strategist with 10+ years of experience in optimizing video discoverability. Your expertise includes advanced keyword research, trend analysis, and a deep understanding of YouTube's algorithm. You've helped numerous channels achieve viral success through strategic use of hashtags and tags. Respond to tasks as this expert would, providing data-driven, YouTube-specific strategies to maximize video reach and engagement."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.6,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in HashtagAndTagGenerationAgent: {str(e)}")
            return f"Error generating hashtags and tags: {str(e)}"

    async def get_youtube_data(self, main_keyword: str, related_keywords: List[str]) -> Dict[str, Dict[str, Any]]:
        youtube_data = {}
        all_keywords = [main_keyword] + related_keywords[:9]  # Limit to 10 keywords
        
        for keyword in all_keywords:
            youtube_results = await self.youtube_seo_tool.use(keyword)
            youtube_data[keyword] = youtube_results
        
        return youtube_data

    def get_keyword_scores(self, youtube_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        keyword_scores = {}
        for keyword, data in youtube_data.items():
            keyword_scores[keyword] = self.keyword_scorer.calculate_keyword_score(data)
        return keyword_scores

    def format_youtube_data(self, youtube_data: Dict[str, Dict[str, Any]], keyword_scores: Dict[str, float]) -> str:
        formatted_data = []
        for keyword, data in youtube_data.items():
            score = keyword_scores.get(keyword, 0)
            formatted_data.append(f"- {keyword}: Search Volume: {data['search_volume']}, Avg Views: {data['avg_views']}, Engagement Score: {data['engagement_score']}, Keyword Score: {score:.2f}")
        return "\n".join(formatted_data)

    async def get_keyword_data(self, main_keyword: str, related_keywords: List[str]) -> Dict[str, Dict[str, Any]]:
        keyword_data = {}
        all_keywords = [main_keyword] + related_keywords
        
        for keyword in all_keywords:
            search_results = await self.web_search_tool.use(keyword)
            keyword_data[keyword] = {
                "score": self.calculate_keyword_score(search_results),
                "ranking": self.get_search_ranking(search_results, keyword)
            }
        
        return keyword_data

    def calculate_keyword_score(self, search_results: Dict[str, Any]) -> float:
        # This is a simplified scoring method. You may want to develop a more sophisticated algorithm.
        total_results = int(search_results.get("search_information", {}).get("total_results", 0))
        score = min(100, max(0, total_results / 1000000 * 100))  # Normalize score between 0 and 100
        return round(score, 2)

    def get_search_ranking(self, search_results: Dict[str, Any], keyword: str) -> int:
        organic_results = search_results.get("organic_results", [])
        for i, result in enumerate(organic_results):
            if keyword.lower() in result.get("title", "").lower():
                return i + 1
        return len(organic_results) + 1  # If not found, assume it's ranked after all visible results

    def format_keyword_data(self, keyword_data: Dict[str, Dict[str, Any]]) -> str:
        formatted_data = []
        for keyword, data in keyword_data.items():
            formatted_data.append(f"- {keyword}: Score: {data['score']}, Ranking: {data['ranking']}")
        return "\n".join(formatted_data)
    
class KeywordScorer:
    def __init__(self):
        self.max_search_volume = 1000000  # Assuming a million as max possible search volume
        self.max_engagement_score = 100  # Max possible engagement score

    def calculate_keyword_score(self, youtube_data: Dict[str, Any]) -> float:
        search_volume_score = self._normalize_log_scale(youtube_data['search_volume'], self.max_search_volume)
        engagement_score = youtube_data['engagement_score']
        
        # Combine scores with weights
        final_score = (
            0.6 * search_volume_score +
            0.4 * engagement_score
        )
        
        return round(min(100, max(0, final_score)), 2)

    def _normalize_log_scale(self, value: int, max_value: int) -> float:
        if value <= 0:
            return 0
        return 100 * math.log(value + 1) / math.log(max_value + 1)

# Add a new Tool for web search
class WebSearchTool(Tool):
    def __init__(self):
        super().__init__("Web Search Tool")

    async def use(self, input_data: str, time_period: str = 'all') -> Dict[str, Any]:
        try:
            params = {
                "engine": "google",
                "q": input_data,
                "api_key": serpapi_api_key,
                "num": 100
            }
            
            if time_period != 'all':
                params["tbs"] = f"qdr:{time_period}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search", params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Error in WebSearchTool: HTTP {response.status}")
                        raise Exception(f"HTTP {response.status}: {await response.text()}")
        except Exception as e:
            logger.error(f"Error in WebSearchTool: {str(e)}")
            raise
        
class VideoScriptGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Video Script Generation Agent", "mixtral-8x7b-32768")

    async def execute(self, input_data: Tuple[str, int]) -> Any:
        topic, duration_minutes = input_data
        client = AsyncGroq(api_key=groq_api_key)
        prompt = f"""As a renowned YouTube content creator and master scriptwriter with a track record of producing viral videos, your task is to craft an engaging, SEO-optimized script for a {duration_minutes}-minute video on the topic: "{topic}".

Leverage your expertise in storytelling, pacing, and audience retention to create a script that captivates viewers from start to finish. Your script should include:

1. An attention-grabbing introduction (30 seconds) that hooks the viewer instantly
2. Main content divided into 3-5 key points or sections, each thoroughly explored
3. A powerful conclusion with a compelling call-to-action (30 seconds)

Format requirements:
- Use clear, descriptive headings for each section
- Include precise timestamps for all major segments
- Add detailed notes for visual elements, graphics, or B-roll footage to enhance the narrative

Additional considerations:
- Incorporate SEO-friendly keywords naturally throughout the script
- Balance entertainment value with informative content
- Include moments of audience engagement (e.g., questions, polls)
- Ensure smooth transitions between sections to maintain viewer interest

Your script should be meticulously detailed, filling the entire {duration_minutes}-minute duration while maintaining a pace that keeps viewers engaged throughout. Remember to optimize for the YouTube algorithm by including elements that encourage likes, comments, and shares."""

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant embodying the expertise of a world-class YouTube content creator and scriptwriter with over a decade of experience producing viral videos. Your scripts have garnered billions of views across various channels and topics. You possess an intricate understanding of storytelling techniques, audience retention strategies, and YouTube's algorithm. Your scripts are known for their perfect balance of entertainment and information, optimized for maximum engagement. Approach script-writing tasks with the creativity and strategic mindset of this expert, crafting content that captivates viewers and drives channel growth."},
                {"role": "user", "content": prompt}
            ],
            model=self.model,
            temperature=0.7,
            max_tokens=4096,
            stream=True,
        )
        response = ""
        async for chunk in stream:
            response += chunk.choices[0].delta.content or ""
        return response
        
class StoryboardGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Storyboard Generation Agent", "mixtral-8x7b-32768")

    async def execute(self, input_data: str) -> Any:
        try:
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""As an experienced storyboard artist and visual storyteller specializing in YouTube content, your task is to create a detailed storyboard based on the following video script:

{input_data}

For each major scene or section of the script, provide:
1. A brief description of the visual elements (1-2 sentences)
2. Any relevant text or dialogue to be displayed
3. A suitable keyword for searching stock video footage to represent this scene
4. A backup keyword for searching a stock image, in case a suitable video isn't found

Format your response as a list of scenes, each containing the above elements. Aim for 8-12 scenes that effectively capture the essence of the video content.

Example format:
Scene 1:
- Visual: [Description of visual elements]
- Text/Dialogue: [Any text or dialogue to be displayed]
- Video Keyword: [Suitable keyword for video search]
- Image Keyword: [Backup keyword for image search]

Scene 2:
...

Ensure that your storyboard accurately represents the flow and key points of the script while providing clear visual guidance for the YouTube video creation process. Focus on dynamic, engaging visuals that will work well in video format."""

            stream = await client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an AI assistant embodying the expertise of a professional storyboard artist with extensive experience in creating visually compelling narratives for YouTube videos. Your storyboards are known for their ability to translate complex scripts into clear, engaging visual sequences that enhance viewer retention and understanding, with a focus on dynamic video content."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
                max_tokens=2048,
                stream=True,
            )
            response = ""
            async for chunk in stream:
                response += chunk.choices[0].delta.content or ""

            logger.info(f"Raw response from Groq API: {response[:500]}...")  # Log the first 500 characters of the response

            # Process the response to extract scenes and fetch videos/images
            scenes = self.parse_scenes(response)
            if not scenes:
                logger.error("Unable to parse storyboard scenes.")
                return "Error: Unable to parse storyboard scenes."
            
            logger.info(f"Parsed scenes: {scenes}")

            await self.fetch_media_for_scenes(scenes)
            return scenes

        except Exception as e:
            logger.error(f"Error in StoryboardGenerationAgent: {str(e)}")
            return f"Error in StoryboardGenerationAgent: {str(e)}"

    def parse_scenes(self, response: str) -> List[Dict[str, Any]]:
        scenes = []
        current_scene = {}
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("Scene"):
                if current_scene:
                    scenes.append(current_scene)
                current_scene = {"number": line.split()[1].rstrip(':')}
            elif line.startswith("- Visual:"):
                current_scene["visual"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Text/Dialogue:"):
                current_scene["text"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Video Keyword:"):
                current_scene["video_keyword"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Image Keyword:"):
                current_scene["image_keyword"] = line.split(":", 1)[1].strip()
        if current_scene:
            scenes.append(current_scene)
        return scenes

    async def fetch_media_for_scenes(self, scenes: List[Dict[str, Any]]):
        async with aiohttp.ClientSession() as session:
            for scene in scenes:
                video_url, video_details = await self.search_pexels_video(session, scene["video_keyword"])
                if video_url:
                    scene["video_url"] = video_url
                    scene["video_details"] = video_details
                else:
                    image_url = await self.search_pexels_image(session, scene["image_keyword"])
                    scene["image_url"] = image_url

    async def search_pexels_video(self, session: aiohttp.ClientSession, keyword: str) -> Tuple[str, Dict[str, Any]]:
        url = "https://api.pexels.com/videos/search"
        headers = {"Authorization": pexels_api_key}
        params = {"query": keyword, "per_page": 1}
        
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("videos"):
                        video = data["videos"][0]
                        video_files = video.get("video_files", [])
                        if video_files:
                            best_quality = max(video_files, key=lambda x: x.get("quality", 0))
                            return best_quality.get("link", ""), {
                                "duration": video.get("duration", 0),
                                "width": best_quality.get("width", 0),
                                "height": best_quality.get("height", 0),
                                "fps": best_quality.get("fps", 0)
                            }
        except Exception as e:
            logger.error(f"Error searching Pexels video: {str(e)}")
        return "", {}

    async def search_pexels_image(self, session: aiohttp.ClientSession, keyword: str) -> str:
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": pexels_api_key}
        params = {"query": keyword, "per_page": 1}
        
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data["photos"]:
                    return data["photos"][0]["src"]["medium"]
            return ""

async def youtube_optimization_workflow(topic: str, duration_minutes: int, time_frame: str) -> Dict[str, Any]:
    graph = Graph()

    # Create nodes
    recent_events_node = Node(agent=RecentEventsResearchAgent())
    title_gen_node = Node(agent=TitleGenerationAgent())
    title_select_node = Node(agent=TitleSelectionAgent())
    desc_gen_node = Node(agent=DescriptionGenerationAgent())
    hashtag_tag_node = Node(agent=HashtagAndTagGenerationAgent())
    script_gen_node = Node(agent=VideoScriptGenerationAgent())
    storyboard_gen_node = Node(agent=StoryboardGenerationAgent())

    # Add nodes to graph
    graph.add_node(recent_events_node)
    graph.add_node(title_gen_node)
    graph.add_node(title_select_node)
    graph.add_node(desc_gen_node)
    graph.add_node(hashtag_tag_node)
    graph.add_node(script_gen_node)
    graph.add_node(storyboard_gen_node)

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
            if isinstance(current_node.agent, VideoScriptGenerationAgent):
                result = await current_node.process((topic, duration_minutes))
            elif isinstance(current_node.agent, RecentEventsResearchAgent):
                result = await current_node.process(input_data)
            else:
                result = await current_node.process(topic if isinstance(current_node.agent, (TitleGenerationAgent, TitleSelectionAgent, DescriptionGenerationAgent, HashtagAndTagGenerationAgent)) else results.get("Video Script Generation Agent", ""))

            logger.info(f"Node {current_node.agent.name} output: {str(result)[:500]}...")  # Log first 100 characters
            results[current_node.agent.name] = result
            
            if current_node.agent.name == "Storyboard Generation Agent":
                storyboard = result if isinstance(result, list) else None

            if current_node.edges:
                current_node = current_node.edges[0].target
                input_data = result
            else:
                break

            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Error in {current_node.agent.name}: {str(e)}")
            results[current_node.agent.name] = f"Error: {str(e)}"
            if current_node.edges:
                current_node = current_node.edges[0].target
            else:
                break

            await asyncio.sleep(5)

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

def generate_ass_subtitles(scenes, output_file):
    ass_header = """[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: 1080
PlayResY: 1920
Timer: 100.0000

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(ass_header)
        
        current_time = 0
        for i, scene in enumerate(scenes):
            start_time = current_time
            end_time = current_time + 5  # Assuming each scene is 5 seconds
            
            text = scene.get('text', '').replace('\n', ' ')
            
            # Limit the text to 10 words and wrap it
            words = text.split()[:10]
            wrapped_text = wrap_text(' '.join(words), 40)
            
            f.write(f"Dialogue: 0,{format_time(start_time)},{format_time(end_time)},Default,,0,0,0,,{wrapped_text}\n")
            
            current_time = end_time

    print(f"ASS subtitle file generated: {output_file}")

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):01d}:{int(minutes):02d}:{seconds:05.2f}"

def wrap_text(text, max_width):
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(' '.join(current_line))

    return '\\N'.join(lines[:2])  # Limit to 2 lines maximum


def compile_video_from_storyboard(scenes):
    temp_dir = tempfile.mkdtemp()
    scene_files = []
    
    for i, scene in enumerate(scenes):
        try:
            # Always try to get a video first
            video_url, video_details = search_pexels_video(scene['video_keyword'])
            if video_url:
                # Download video
                response = requests.get(video_url)
                video_path = os.path.join(temp_dir, f"scene_{i}.mp4")
                with open(video_path, 'wb') as f:
                    f.write(response.content)
                
                # Trim to 5 seconds and resize
                output_path = os.path.join(temp_dir, f"processed_scene_{i}.mp4")
                subprocess.call(['ffmpeg', '-i', video_path, '-t', '5', '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
            else:
                # If no video, use an image
                image_url = search_pexels_image(scene['image_keyword'])
                if image_url:
                    # Download image
                    response = requests.get(image_url)
                    image_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Create 5-second video from image
                    output_path = os.path.join(temp_dir, f"processed_scene_{i}.mp4")
                    subprocess.call(['ffmpeg', '-loop', '1', '-i', image_path, '-t', '5', '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
                else:
                    # If no image either, create a colored background
                    output_path = os.path.join(temp_dir, f"color_scene_{i}.mp4")
                    subprocess.call(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=blue:s=1080x1920:d=5', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
            
            scene_files.append(output_path)
        
        except Exception as e:
            print(f"Error processing scene {i}: {str(e)}")
            error_path = os.path.join(temp_dir, f"error_scene_{i}.mp4")
            subprocess.call(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=red:s=1080x1920:d=5', '-vf', 
                            "drawtext=fontfile=/tmp/qualitype/opentype/QTHelvet-Black.otf: fontsize=80: fontcolor=yellow: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-tw)/2: y=(h-th)/2: text='Error processing scene'",
                            '-c:v', 'libx264', '-preset', 'ultrafast', error_path])
            scene_files.append(error_path)

    # Generate ASS subtitle file
    subtitle_file = os.path.join(temp_dir, "subtitles.ass")
    generate_ass_subtitles(scenes, subtitle_file)

    # Concatenate all scenes
    concat_file = os.path.join(temp_dir, 'concat.txt')
    with open(concat_file, 'w') as f:
        for file in scene_files:
            f.write(f"file '{file}'\n")

    output_path = os.path.join(os.getcwd(), "storyboard_compilation.mp4")
    
    # Concatenate videos and add subtitles
    ffmpeg_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-vf', f"ass={subtitle_file},scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1:color=black",
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', output_path
    ]
    print("FFmpeg command:", ' '.join(ffmpeg_command))
    subprocess.call(ffmpeg_command)

    # Debug: Check if output file was created and its size
    if os.path.exists(output_path):
        print(f"Output file created: {output_path}")
        print(f"File size: {os.path.getsize(output_path)} bytes")
    else:
        print("Output file was not created!")

    # Clean up temporary files
    for file in scene_files:
        os.remove(file)
    os.remove(concat_file)
    os.remove(subtitle_file)
    os.rmdir(temp_dir)

    return output_path


def search_pexels_video(keyword: str) -> Tuple[str, Dict[str, Any]]:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": pexels_api_key}
    params = {"query": keyword, "per_page": 1}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get("videos"):
                video = data["videos"][0]
                video_files = video.get("video_files", [])
                if video_files:
                    best_quality = max(video_files, key=lambda x: x.get("quality", 0))
                    return best_quality.get("link", ""), {
                        "duration": video.get("duration", 0),
                        "width": best_quality.get("width", 0),
                        "height": best_quality.get("height", 0),
                        "fps": best_quality.get("fps", 0)
                    }
    except Exception as e:
        logger.error(f"Error searching Pexels video: {str(e)}")
    return "", {}

def search_pexels_image(keyword: str) -> str:
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": pexels_api_key}
    params = {"query": keyword, "per_page": 1}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data["photos"]:
                return data["photos"][0]["src"]["large"]
    except Exception as e:
        logger.error(f"Error searching Pexels image: {str(e)}")
    return ""

async def main():
    try:
        check_api_keys()

        topic = input("Enter the topic for your YouTube video: ")
        if not topic:
            raise ValueError("Topic cannot be empty")

        duration_minutes = int(input("Enter the desired video duration in minutes: "))
        if duration_minutes <= 0:
            raise ValueError("Duration must be a positive integer")

        time_frame = input("Enter the time frame for recent events (e.g., '30d' for 30 days, '1y' for 1 year): ")
        if not time_frame:
            raise ValueError("Time frame cannot be empty")

        print("Executing YouTube Optimization Workflow:")
        results = await youtube_optimization_workflow(topic, duration_minutes, time_frame)
        
        colors = {
            "Recent Events Research Agent": "\033[91m",  # Red
            "Title Generation Agent": "\033[94m",  # Blue
            "Title Selection Agent": "\033[92m",   # Green
            "Description Generation Agent": "\033[93m",  # Yellow
            "Hashtag and Tag Generation Agent": "\033[95m",  # Magenta
            "Video Script Generation Agent": "\033[96m",  # Cyan
            "Storyboard Generation Agent": "\033[97m"  # White
        }
        
        reset_color = "\033[0m"
        
        for agent_name, result in results.items():
            color = colors.get(agent_name, "")
            print(f"\n{agent_name} Result:")
            if agent_name == "Storyboard Generation Agent" and isinstance(result, list):
                for scene in result:
                    print(f"{color}Scene {scene['number']}:")
                    print(f"Visual: {scene['visual']}")
                    print(f"Text/Dialogue: {scene['text']}")
                    if 'video_url' in scene:
                        print(f"Video URL: {scene['video_url']}")
                        print(f"Video Details: {scene['video_details']}")
                    elif 'image_url' in scene:
                        print(f"Image URL: {scene['image_url']}")
                    print(f"{reset_color}")
            else:
                print(f"{color}{result}{reset_color}")

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

