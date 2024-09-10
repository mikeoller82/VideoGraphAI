# plugins/agents.py
from abc import ABC, abstractmethod
from utils.caching import cache_result
from typing import Any, List, Dict, Tuple
from groq import AsyncGroq
from plugins.tools import WebSearchTool, YouTubeSEOTool, KeywordScorer
import os
import aiohttp
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")

# Helper functions
def check_api_keys():
    required_keys = ["GROQ_API_KEY", "SERPAPI_API_KEY", "YOUTUBE_API_KEY", "PEXELS_API_KEY"]
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required API key: {key}")
              
async def get_data(query: str) -> List[Dict[str, Any]]:
    groq = AsyncGroq(api_key=groq_api_key)
    data = await groq.query(query)
    return data


class Agent(ABC):
    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model

    @abstractmethod
    @cache_result
    async def execute(self, input_data: Any) -> Any:
        pass

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
    name = "Recent Events Research Agent"

    def __init__(self):
        super().__init__("Recent Events Research Agent", "llama-3.1-8b-instant")
        self.web_search_tool = WebSearchTool()

    async def execute(self, input_data: Dict[str, Any] = None) -> Any:
        if input_data is None:
            raise ValueError("Input data is required")
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
        self.youtube_seo_tool = YouTubeSEOTool(api_key=youtube_api_key)
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
