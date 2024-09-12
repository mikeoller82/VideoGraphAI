import asyncio
import aiohttp
import tempfile
import subprocess
import json
import logging
import shutil
from dotenv import load_dotenv
import os
from moviepy.editor import *
from moviepy.video.fx.all import crop
import moviepy.config as conf
conf.change_settings({"IMAGEMAGICK_BINARY": r"/usr/bin/convert"})
from moviepy.editor import ColorClip, TextClip, CompositeVideoClip, concatenate_videoclips, VideoFileClip
import requests
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
from groq import AsyncGroq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
serpapi_api_key = os.getenv("SERPER_API_KEY")


# Helper functions
async def get_data(query: str) -> List[Dict[str, Any]]:
    groq = AsyncGroq(api_key=groq_api_key)
    data = await groq.query(query)
    return data

def check_api_keys():
    required_keys = ["GROQ_API_KEY", "PEXELS_API_KEY", "ELEVENLABS_API_KEY"]
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
    
class VoiceModule(ABC):

    def __init__(self):
        pass
    @abstractmethod    
    def update_usage(self):
        pass

    @abstractmethod
    def get_remaining_characters(self):
        pass

    @abstractmethod
    def generate_voice(self,text, outputfile):
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
        
class ElevenLabsAPI:

    def __init__(self, api_key):
        self.api_key = api_key
        self.url_base = 'https://api.elevenlabs.io/v1/'
        self.get_voices()

    def get_voices(self):
        '''Get the list of voices available'''
        url = self.url_base + 'voices'
        headers = {'accept': 'application/json'}
        if self.api_key:
            headers['xi-api-key'] = self.api_key
        response = requests.get(url, headers=headers)
        self.voices = {voice['name']: voice['voice_id'] for voice in response.json()['voices']}
        return self.voices

    def get_remaining_characters(self):
        '''Get the number of characters remaining'''
        url = self.url_base + 'user'
        headers = {'accept': '*/*', 'xi-api-key': self.api_key, 'Content-Type': 'application/json'}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            sub = response.json()['subscription']
            return sub['character_limit'] - sub['character_count']
        else:
            raise Exception(response.json()['detail']['message'])

    def generate_voice(self, text, character, filename, stability=0.2, clarity=0.1):
        '''Generate a voice'''
        if character not in self.voices:
            print(character, 'is not in the array of characters: ', list(self.voices.keys()))

        voice_id = self.voices[character]
        url = f'{self.url_base}text-to-speech/{voice_id}/stream'
        headers = {'accept': '*/*', 'xi-api-key': self.api_key, 'Content-Type': 'application/json'}
        data = json.dumps({"model_id": "eleven_multilingual_v2", "text": text, "stability": stability, "similarity_boost": clarity})
        response = requests.post(url, headers=headers, data=data)

        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
                return filename
        else:
            message = response.text
            raise Exception(f'Error in response, {response.status_code} , message: {message}')
        
class ElevenLabsVoiceModule(VoiceModule):
    def __init__(self, api_key, voiceName, checkElevenCredits=False):
        self.api_key = api_key
        self.voiceName = voiceName
        self.remaining_credits = None
        self.eleven_labs_api = ElevenLabsAPI(self.api_key)
        self.update_usage()
        if checkElevenCredits and self.get_remaining_characters() < 1200:
            raise Exception(f"Your ElevenLabs API KEY doesn't have enough credits ({self.remaining_credits} character remaining). Minimum required: 1200 characters (equivalent to a 45sec short)")
        super().__init__()

    def generate_voice(self, text, outputfile):
        if self.get_remaining_characters() >= len(text):
            voice_id = self.eleven_labs_api.voices[self.voiceName]
            url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream'
            headers = {'accept': '*/*', 'xi-api-key': self.api_key, 'Content-Type': 'application/json'}
            data = json.dumps({"model_id": "eleven_multilingual_v2", "text": text})
            response = requests.post(url, headers=headers, data=data)

            if response.status_code == 200:
                with open(outputfile, 'wb') as f:
                    f.write(response.content)
                self.update_usage()
                return outputfile
            else:
                raise Exception(f"Error generating voice: {response.text}")
        else:
            raise Exception(f"You cannot generate {len(text)} characters as your ElevenLabs key has only {self.remaining_credits} characters remaining")

    def update_usage(self):
        self.remaining_credits = self.eleven_labs_api.get_remaining_characters()
        return self.remaining_credits

    def get_remaining_characters(self):
        return self.remaining_credits if self.remaining_credits else self.eleven_labs_api.get_remaining_characters()

    
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

# Updated AI Agents for YouTube content optimization
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

class HashtagAndTagGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Hashtag and Tag Generation Agent", "llama-3.1-70b-versatile")

    async def execute(self, input_data: str) -> Any:
        try:
            client = AsyncGroq(api_key=groq_api_key)
            prompt = f"""As a leading YouTube SEO specialist and social media strategist with a proven track record in optimizing video discoverability and virality, your task is to create an engaging and relevant set of hashtags and tags for the YouTube video titled "{input_data}". Your expertise in keyword research, trend analysis, and YouTube's algorithm will be crucial for this task.

Develop the following:

1. 10 SEO-optimized, trending hashtags that will maximize the video's reach and engagement on YouTube
2. 35 high-value SEO tags, combining keywords strategically to boost the video's search ranking on YouTube

In your selection process, prioritize:
- Relevance to the video title and content
- Potential search volume on YouTube
- Engagement potential (views, likes, comments)
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

class VideoScriptGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Video Script Generation Agent", "mixtral-8x7b-32768")

    async def execute(self, input_data: str) -> Any:
        client = AsyncGroq(api_key=groq_api_key)
        prompt = f"""As a YouTube Shorts content creator, craft a brief, engaging script for a 60-second vertical video on the topic: "{input_data}".

Your script should include:
1. An attention-grabbing opening (5-10 seconds)
2. 2-3 key points or facts (40-45 seconds)
3. A strong call-to-action conclusion (5-10 seconds)

Format the script with clear timestamps and keep each segment concise for a fast-paced Short. Optimize for viewer retention and engagement."""

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in creating viral YouTube Shorts scripts."},
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


class StoryboardGenerationAgent(Agent):
    def __init__(self):
        super().__init__("Storyboard Generation Agent", "mixtral-8x7b-32768")

    async def execute(self, input_data: str) -> Any:
        client = AsyncGroq(api_key=groq_api_key)
        prompt = f"""Create a storyboard for a 60-second YouTube Short based on this script:

{input_data}

For each major scene (aim for 4-6 scenes), provide:
1. Visual: A brief description of the visual elements (1 sentence)
2. Text: The exact text/dialogue for voiceover and subtitles
3. Video Keyword: A suitable keyword for searching stock video footage
4. Image Keyword: A backup keyword for searching a stock image

Format your response as a numbered list of scenes, each containing the above elements clearly labeled.

Example:
1. Visual: A person looking confused at a complex math equation on a chalkboard
   Text: "Have you ever felt overwhelmed by math?"
   Video Keyword: student struggling with math
   Image Keyword: confused face mathematics

2. Visual: ...
   Text: ...
   Video Keyword: ...
   Image Keyword: ...

Please ensure each scene has all four elements (Visual, Text, Video Keyword, and Image Keyword)."""

        stream = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI assistant specializing in creating detailed storyboards for YouTube Shorts."},
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

        logger.info(f"Raw storyboard response: {response}")
        scenes = self.parse_scenes(response)
        if not scenes:
            raise ValueError("Failed to generate valid storyboard scenes")
        await self.fetch_media_for_scenes(scenes)
        return scenes

    def parse_scenes(self, response: str) -> List[Dict[str, Any]]:
        scenes = []
        current_scene = {}
        scene_number = 0

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                if current_scene:
                    scenes.append(self.validate_and_fix_scene(current_scene, scene_number))
                scene_number = int(line.split('.')[0])
                current_scene = {"number": scene_number}
            elif ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key == 'text':
                    current_scene['narration_text'] = value
                elif key in ['visual', 'video keyword', 'image keyword']:
                    current_scene[key.replace(' ', '_')] = value

        if current_scene:
            scenes.append(self.validate_and_fix_scene(current_scene, scene_number))

        logger.info(f"Parsed and validated scenes: {scenes}")
        return scenes

    def validate_and_fix_scene(self, scene: Dict[str, Any], scene_number: int) -> Dict[str, Any]:
        required_keys = ['visual', 'narration_text', 'video_keyword', 'image_keyword']
        for key in required_keys:
            if key not in scene:
                if key == 'visual':
                    scene[key] = f"Visual representation of scene {scene_number}"
                elif key == 'narration_text':
                    scene[key] = ""  # Empty string if no narration
                elif key == 'video_keyword':
                    scene[key] = f"video scene {scene_number}"
                elif key == 'image_keyword':
                    scene[key] = f"image scene {scene_number}"
                logger.warning(f"Added missing {key} for scene {scene_number}")
        return scene

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
        
    def fallback_scene_generation(self, invalid_scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid_scenes = []
        for scene in invalid_scenes:
            if 'visual' not in scene:
                scene['visual'] = f"Visual representation of: {scene.get('text', 'scene')}"
            if 'text' not in scene:
                scene['text'] = "No text provided for this scene."
            if 'video_keyword' not in scene:
                scene['video_keyword'] = scene.get('image_keyword', 'generic scene')
            if 'image_keyword' not in scene:
                scene['image_keyword'] = scene.get('video_keyword', 'generic image')
            valid_scenes.append(scene)
        return valid_scenes

async def youtube_shorts_workflow(topic: str) -> Dict[str, Any]:
    graph = Graph()

    script_gen_node = Node(agent=VideoScriptGenerationAgent())
    storyboard_gen_node = Node(agent=StoryboardGenerationAgent())
    graph.add_node(script_gen_node)
    graph.add_node(storyboard_gen_node)
    graph.add_edge(Edge(script_gen_node, storyboard_gen_node))

    current_node = script_gen_node
    results = {}

    while current_node:
        try:
            result = await current_node.process(topic if isinstance(current_node.agent, VideoScriptGenerationAgent) else results.get("Video Script Generation Agent", ""))
            results[current_node.agent.name] = result
            
            if current_node.edges:
                current_node = current_node.edges[0].target
            else:
                break

            await asyncio.sleep(2)

        except Exception as e:
            logger.error(f"Error in {current_node.agent.name}: {str(e)}")
            results[current_node.agent.name] = f"Error: {str(e)}"
            break

    try:
        storyboard = results.get("Storyboard Generation Agent", [])
        if isinstance(storyboard, str) and storyboard.startswith("Error"):
            raise ValueError(storyboard)
        if not isinstance(storyboard, list) or len(storyboard) == 0:
            raise ValueError("Invalid storyboard generated")
        
        output_path = compile_youtube_short(storyboard)
        print(f"YouTube Short saved as '{output_path}'")
    except Exception as e:
        logger.error(f"Error compiling YouTube Short: {str(e)}")
        print(f"Error compiling YouTube Short: {str(e)}")

    return results

def compile_youtube_short(scenes):
    temp_dir = tempfile.mkdtemp()
    scene_files = []
    subtitle_file = os.path.join(temp_dir, "subtitles.srt")
    audio_file = os.path.join(temp_dir, "voiceover.mp3")
    
    generate_subtitles(scenes, subtitle_file)
    generate_voiceover(scenes, audio_file)
    
    # Verify that the audio file exists
    if not os.path.exists(audio_file):
        logger.error(f"Audio file not found: {audio_file}")
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    for i, scene in enumerate(scenes):
        try:
            video_url = scene.get('video_url')
            if video_url:
                # Download and process video
                response = requests.get(video_url)
                video_path = os.path.join(temp_dir, f"scene_{i}.mp4")
                with open(video_path, 'wb') as f:
                    f.write(response.content)
                
                # Trim and resize for vertical Short
                output_path = os.path.join(os.getcwd(), "youtube_short.mp4")
                duration = 60 / len(scenes)  # Distribute time equally among scenes
                subprocess.call(['ffmpeg', '-i', video_path, '-t', str(duration), '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
                scene_files.append(output_path)
            else:
                # Use image if available, otherwise create colored background
                image_url = scene.get('image_url')
                if image_url:
                    response = requests.get(image_url)
                    image_path = os.path.join(temp_dir, f"scene_{i}.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(response.content)
                    output_path = os.path.join(temp_dir, f"processed_scene_{i}.mp4")
                    duration = 60 / len(scenes)
                    subprocess.call(['ffmpeg', '-loop', '1', '-i', image_path, '-t', str(duration), '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
                else:
                    output_path = os.path.join(temp_dir, f"color_scene_{i}.mp4")
                    duration = 60 / len(scenes)
                    subprocess.call(['ffmpeg', '-f', 'lavfi', '-i', f'color=c=blue:s=1080x1920:d={duration}', '-c:v', 'libx264', '-preset', 'ultrafast', output_path])
                scene_files.append(output_path)
        
        except Exception as e:
            logger.error(f"Error processing scene {i}: {str(e)}")
            # Create error scene
            error_path = os.path.join(temp_dir, f"error_scene_{i}.mp4")
            subprocess.call(['ffmpeg', '-f', 'lavfi', '-i', f'color=c=red:s=1080x1920:d={60/len(scenes)}', '-vf', 
                            "drawtext=fontfile=/path/to/font.ttf: fontsize=80: fontcolor=white: box=1: boxcolor=black@0.5: boxborderw=5: x=(w-tw)/2: y=(h-th)/2: text='Error in scene'",
                            '-c:v', 'libx264', '-preset', 'ultrafast', error_path])
            scene_files.append(error_path)

    # Concatenate all scenes
    concat_file = os.path.join(temp_dir, 'concat.txt')
    with open(concat_file, 'w') as f:
        for file in scene_files:
            f.write(f"file '{file}'\n")

    output_path = os.path.join(os.getcwd(), "youtube_short.mp4")
    
    # Concatenate videos, add subtitles and voiceover
    ffmpeg_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-i', audio_file,
        '-vf', f"subtitles={subtitle_file}",
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', output_path
    ]
    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info(f"FFmpeg command output: {result.stdout}")
        logger.info(f"YouTube Short compiled successfully: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error compiling YouTube Short: {e.stderr}")
        raise

    # Clean up
    for file in scene_files:
        try:
            os.remove(file)
        except Exception as e:
            logger.warning(f"Error removing file {file}: {str(e)}")
    
    try:
        os.remove(concat_file)
        os.remove(subtitle_file)
        os.remove(audio_file)
    except Exception as e:
        logger.warning(f"Error removing temporary files: {str(e)}")

    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"Error removing temporary directory {temp_dir}: {str(e)}")

    return output_path

def generate_subtitles(scenes, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        start_time = 0
        for i, scene in enumerate(scenes):
            end_time = start_time + (60 / len(scenes))
            text = scene.get('narration_text', '').replace('\n', ' ')
            if text:
                f.write(f"{i+1}\n")
                f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
                f.write(f"{text}\n\n")
            start_time = end_time

def generate_voiceover(scenes, output_file):
    voice_module = ElevenLabsVoiceModule(elevenlabs_api_key, "Brian")
    full_text = " ".join([scene.get('narration_text', '') for scene in scenes])
    logger.info(f"Generating voiceover for text of length: {len(full_text)}")
    try:
        voice_module.generate_voice(full_text, output_file)
        logger.info(f"Voiceover generated and saved to {output_file}")
        if os.path.exists(output_file):
            logger.info(f"Voiceover file size: {os.path.getsize(output_file)} bytes")
        else:
            logger.error(f"Voiceover file not found at {output_file}")
    except Exception as e:
        logger.error(f"Error generating voiceover: {str(e)}")
        raise

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

async def main():
    try:
        check_api_keys()
        topic = input("Enter the topic for your YouTube Short: ")
        if not topic:
            raise ValueError("Topic cannot be empty")

        print("Executing YouTube Shorts Workflow:")
        results = await youtube_shorts_workflow(topic)
        
        for agent_name, result in results.items():
            print(f"\n{agent_name} Result:")
            if isinstance(result, list):
                for scene in result:
                    print(json.dumps(scene, indent=2))
            else:
                print(result)

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())