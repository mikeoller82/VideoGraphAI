import asyncio
import aiohttp
import logging
import json
import requests
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
from abc import ABC, abstractmethod
from utils.caching import cache_result

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nltk.download('punkt')
nltk.download('stopwords')

serpapi_api_key = os.getenv("SERPAPI_API_KEY")
pexels_api_key = os.getenv("PEXELS_API_KEY")



class Tool(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    @cache_result
    async def use(self, input_data: Any) -> Any:
        pass

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


class YouTubeSEOTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        self.stop_words = set(stopwords.words('english'))
        self.keyword = None
        
    async def use(self, keyword: str) -> None:
        self.keyword = keyword
        await self._analyze_keyword(keyword)
          

    async def _analyze_keyword(self, keyword: str) -> Dict[str, Any]:
        search_data = await self._search_videos(keyword)
        video_ids = [item['id']['videoId'] for item in search_data['items']]
        video_details = await self._get_video_details(video_ids)
        
        analysis = {
            "keyword": keyword,
            "search_volume": self._estimate_search_volume(search_data),
            "competition": self._calculate_competition(video_details),
            "trending_score": self._calculate_trending_score(video_details),
            "top_tags": self._extract_top_tags(video_details),
            "optimal_title_length": self._calculate_optimal_title_length(video_details),
            "optimal_description_length": self._calculate_optimal_description_length(video_details),
            "best_upload_times": self._analyze_upload_times(video_details),
            "engagement_rate": self._calculate_engagement_rate(video_details),
            "keyword_difficulty": self._calculate_keyword_difficulty(video_details),
            "related_keywords": await self._find_related_keywords(keyword),
            "sentiment_analysis": self._analyze_sentiment(video_details),
            "content_gaps": self._identify_content_gaps(video_details),
        }
        
        return analysis

    async def _search_videos(self, keyword: str) -> Dict[str, Any]:
        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": 50,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()

    async def _get_video_details(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/videos"
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['items']

    def _estimate_search_volume(self, search_data: Dict[str, Any]) -> int:
        return int(search_data['pageInfo']['totalResults'])

    def _calculate_competition(self, video_details: List[Dict[str, Any]]) -> float:
        total_views = sum(int(video['statistics']['viewCount']) for video in video_details)
        avg_views = total_views / len(video_details)
        max_views = max(int(video['statistics']['viewCount']) for video in video_details)
        return 1 - (avg_views / max_views)

    def _calculate_trending_score(self, video_details: List[Dict[str, Any]]) -> float:
        now = datetime.utcnow()
        scores = []
        for video in video_details:
            published_at = datetime.strptime(video['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
            days_since_published = (now - published_at).days
            if days_since_published == 0:
                days_since_published = 1
            views = int(video['statistics']['viewCount'])
            score = views / days_since_published
            scores.append(score)
        return sum(scores) / len(scores)

    def _extract_top_tags(self, video_details: List[Dict[str, Any]]) -> List[str]:
        all_tags = []
        for video in video_details:
            all_tags.extend(video['snippet'].get('tags', []))
        return [tag for tag, _ in Counter(all_tags).most_common(10)]

    def _calculate_optimal_title_length(self, video_details: List[Dict[str, Any]]) -> int:
        title_lengths = [len(video['snippet']['title']) for video in video_details]
        return int(sum(title_lengths) / len(title_lengths))

    def _calculate_optimal_description_length(self, video_details: List[Dict[str, Any]]) -> int:
        desc_lengths = [len(video['snippet']['description']) for video in video_details]
        return int(sum(desc_lengths) / len(desc_lengths))

    def _analyze_upload_times(self, video_details: List[Dict[str, Any]]) -> List[str]:
        upload_times = [datetime.strptime(video['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").strftime("%A %H:00")
                        for video in video_details]
        return [time for time, _ in Counter(upload_times).most_common(3)]

    def _calculate_engagement_rate(self, video_details: List[Dict[str, Any]]) -> float:
        rates = []
        for video in video_details:
            views = int(video['statistics']['viewCount'])
            likes = int(video['statistics'].get('likeCount', 0))
            dislikes = int(video['statistics'].get('dislikeCount', 0))
            comments = int(video['statistics'].get('commentCount', 0))
            rate = (likes + dislikes + comments) / views if views > 0 else 0
            rates.append(rate)
        return sum(rates) / len(rates)

    def _calculate_keyword_difficulty(self, video_details: List[Dict[str, Any]]) -> float:
        channel_ids = set(video['snippet']['channelId'] for video in video_details)
        unique_channels = len(channel_ids)
        total_channels = len(video_details)
        return unique_channels / total_channels

    async def _find_related_keywords(self, keyword: str) -> List[str]:
        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "relatedToVideoId": await self._get_top_video_id(keyword),
            "type": "video",
            "maxResults": 50,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                titles = [item['snippet']['title'] for item in data['items']]
                return self._extract_keywords(titles)

    async def _get_top_video_id(self, keyword: str) -> str:
        search_data = await self._search_videos(keyword)
        return search_data['items'][0]['id']['videoId']

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        words = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            words.extend([word for word in tokens if word.isalnum() and word not in self.stop_words])
        return [word for word, _ in Counter(words).most_common(10)]

    def _analyze_sentiment(self, video_details: List[Dict[str, Any]]) -> str:
        positive_words = set(['amazing', 'awesome', 'best', 'fantastic', 'great', 'love', 'perfect', 'wonderful'])
        negative_words = set(['awful', 'bad', 'hate', 'horrible', 'terrible', 'worst'])
        
        sentiment_scores = []
        for video in video_details:
            text = video['snippet']['title'] + " " + video['snippet']['description']
            words = word_tokenize(text.lower())
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            sentiment_scores.append(positive_count - negative_count)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        if avg_sentiment > 2:
            return "Positive"
        elif avg_sentiment < -2:
            return "Negative"
        else:
            return "Neutral"

    def _identify_content_gaps(self, video_details: List[Dict[str, Any]]) -> List[str]:
        all_words = []
        for video in video_details:
            text = video['snippet']['title'] + " " + video['snippet']['description']
            words = word_tokenize(text.lower())
            all_words.extend([word for word in words if word.isalnum() and word not in self.stop_words])
        
        word_freq = Counter(all_words)
        total_words = sum(word_freq.values())
        word_importance = {}
        for word, freq in word_freq.items():
            tf = freq / total_words
            idf = math.log(len(video_details) / sum(1 for video in video_details if word in video['snippet']['title'] + " " + video['snippet']['description']))
            word_importance[word] = tf * idf
        
        return [word for word, _ in sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:5]]

    async def analyze_video(self, video_id: str, keyword: str) -> Dict[str, Any]:
        self.keyword = keyword  # Store the keyword as an instance variable
        video_details = await self._get_video_details([video_id])
        if not video_details:
            return {"error": "Video not found"}
        
        video = video_details[0]
        analysis = {
            "title_score": self._score_title(video['snippet']['title']),
            "description_score": self._score_description(video['snippet']['description']),
            "tag_score": self._score_tags(video['snippet'].get('tags', [])),
            "thumbnail_score": await self._score_thumbnail(video_id),
            "engagement_score": self._score_engagement(video['statistics']),
            "optimization_tips": self._generate_optimization_tips(video)
        }
        
        return analysis

    def _score_title(self, title: str) -> float:
        length_score = 1 - abs(len(title) - 60) / 60  # 60 characters is considered optimal
        keyword_score = sum(1 for word in self.keyword.split() if word.lower() in title.lower()) / len(self.keyword.split())
        return (length_score + keyword_score) / 2

    def _score_description(self, description: str) -> float:
        length_score = min(len(description) / 5000, 1)  # 5000 characters is considered optimal
        keyword_score = sum(1 for word in self.keyword.split() if word.lower() in description.lower()) / len(self.keyword.split())
        return (length_score + keyword_score) / 2

    def _score_tags(self, tags: List[str]) -> float:
        if not tags:
            return 0
        tag_count_score = min(len(tags) / 15, 1)  # 15 tags is considered optimal
        keyword_score = sum(1 for tag in tags if self.keyword.lower() in tag.lower()) / len(tags)
        return (tag_count_score + keyword_score) / 2

    async def _score_thumbnail(self, video_id: str) -> float:
        # This would ideally involve image analysis, but for simplicity, we'll assume all thumbnails are equally good
        return 1.0

    def _score_engagement(self, statistics: Dict[str, str]) -> float:
        views = int(statistics['viewCount'])
        likes = int(statistics.get('likeCount', 0))
        dislikes = int(statistics.get('dislikeCount', 0))
        comments = int(statistics.get('commentCount', 0))
        
        like_ratio = likes / (likes + dislikes) if likes + dislikes > 0 else 0
        comment_ratio = comments / views if views > 0 else 0
        
        return (like_ratio + comment_ratio) / 2

    def _generate_optimization_tips(self, video: Dict[str, Any]) -> List[str]:
        tips = []
        
        if len(video['snippet']['title']) < 30:
            tips.append("Consider making your title longer to include more keywords")
        
        if len(video['snippet']['description']) < 250:
            tips.append("Your description is quite short. Consider adding more detailed information")
        
        if 'tags' not in video['snippet'] or len(video['snippet']['tags']) < 10:
            tips.append("Add more tags to improve discoverability")
        
        if int(video['statistics'].get('commentCount', 0)) < 10:
            tips.append("Encourage more comments to boost engagement")
        
        return tips

    async def analyze_channel(self, channel_id: str) -> Dict[str, Any]:
        channel_data = await self._get_channel_data(channel_id)
        video_ids = await self._get_channel_video_ids(channel_id)
        video_details = await self._get_video_details(video_ids[:50])  # Analyze up to 50 most recent videos
        
        analysis = {
            "subscriber_count": int(channel_data['statistics']['subscriberCount']),
            "view_count": int(channel_data['statistics']['viewCount']),
            "video_count": int(channel_data['statistics']['videoCount']),
            "avg_views_per_video": self._calculate_avg_views(video_details),
            "engagement_rate": self._calculate_channel_engagement_rate(video_details),
            "upload_frequency": self._calculate_upload_frequency(video_details),
            "top_performing_videos": self._get_top_performing_videos(video_details),
            "keyword_analysis": self._analyze_channel_keywords(video_details),
            "improvement_suggestions": self._generate_channel_suggestions(channel_data, video_details)
        }
        
        return analysis

    async def _get_channel_data(self, channel_id: str) -> Dict[str, Any]:
        url = f"{self.base_url}/channels"
        params = {
            "part": "snippet,statistics",
            "id": channel_id,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['items'][0]

    async def _get_channel_video_ids(self, channel_id: str) -> List[str]:
        url = f"{self.base_url}/search"
        params = {
            "part": "id",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "maxResults": 50,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return [item['id']['videoId'] for item in data['items']]

    def _calculate_avg_views(self, video_details: List[Dict[str, Any]]) -> float:
        total_views = sum(int(video['statistics']['viewCount']) for video in video_details)
        return total_views / len(video_details)

    def _calculate_channel_engagement_rate(self, video_details: List[Dict[str, Any]]) -> float:
        engagement_rates = []
        for video in video_details:
            views = int(video['statistics']['viewCount'])
            likes = int(video['statistics'].get('likeCount', 0))
            dislikes = int(video['statistics'].get('dislikeCount', 0))
            comments = int(video['statistics'].get('commentCount', 0))
            if views > 0:
                engagement_rates.append((likes + dislikes + comments) / views)
        return sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0

    def _calculate_upload_frequency(self, video_details: List[Dict[str, Any]]) -> str:
        if len(video_details) < 2:
            return "Insufficient data"
        
        publish_dates = [datetime.strptime(video['snippet']['publishedAt'], "%Y-%m-%dT%H:%M:%SZ") for video in video_details]
        publish_dates.sort(reverse=True)
        
        time_diffs = [(publish_dates[i] - publish_dates[i+1]).days for i in range(len(publish_dates)-1)]
        avg_days_between_uploads = sum(time_diffs) / len(time_diffs)
        
        if avg_days_between_uploads < 1:
            return f"Multiple times per day (every {avg_days_between_uploads:.2f} days)"
        elif avg_days_between_uploads < 7:
            return f"{7/avg_days_between_uploads:.1f} times per week"
        elif avg_days_between_uploads < 30:
            return f"{30/avg_days_between_uploads:.1f} times per month"
        else:
            return f"Every {avg_days_between_uploads:.1f} days"

    def _get_top_performing_videos(self, video_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sorted_videos = sorted(video_details, key=lambda x: int(x['statistics']['viewCount']), reverse=True)
        return [{"title": video['snippet']['title'], "views": int(video['statistics']['viewCount'])} for video in sorted_videos[:5]]

    def _analyze_channel_keywords(self, video_details: List[Dict[str, Any]]) -> Dict[str, int]:
        all_words = []
        for video in video_details:
            text = video['snippet']['title'] + " " + video['snippet']['description']
            words = word_tokenize(text.lower())
            all_words.extend([word for word in words if word.isalnum() and word not in self.stop_words])
        return dict(Counter(all_words).most_common(10))

    def _generate_channel_suggestions(self, channel_data: Dict[str, Any], video_details: List[Dict[str, Any]]) -> List[str]:
        suggestions = []
        
        # Check upload frequency
        upload_frequency = self._calculate_upload_frequency(video_details)
        if "month" in upload_frequency or float(upload_frequency.split()[1]) > 14:
            suggestions.append("Consider uploading more frequently to maintain audience engagement")
        
        # Check video length consistency
        durations = [self._parse_duration(video['contentDetails']['duration']) for video in video_details]
        avg_duration = sum(durations) / len(durations)
        duration_variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
        if duration_variance > 100:
            suggestions.append("Your video lengths are inconsistent. Consider standardizing your content length")
        
        # Check thumbnail consistency
        if not self._check_thumbnail_consistency(video_details):
            suggestions.append("Your video thumbnails lack consistency. Consider creating a consistent thumbnail style")
        
        # Check description length
        avg_desc_length = sum(len(video['snippet']['description']) for video in video_details) / len(video_details)
        if avg_desc_length < 100:
            suggestions.append("Your video descriptions are quite short. Consider adding more detailed information")
        
        # Check for end screens and cards
        if not self._check_end_screens_and_cards(video_details):
            suggestions.append("Consider using end screens and cards to promote your other videos and increase watch time")
        
        return suggestions

    def _parse_duration(self, duration: str) -> int:
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
        hours = int(match.group(1)[:-1]) if match.group(1) else 0
        minutes = int(match.group(2)[:-1]) if match.group(2) else 0
        seconds = int(match.group(3)[:-1]) if match.group(3) else 0
        return hours * 3600 + minutes * 60 + seconds

    def _check_thumbnail_consistency(self, video_details: List[Dict[str, Any]]) -> bool:
        # This is a placeholder. In a real implementation, you'd use image analysis to check for consistent branding, color schemes, etc.
        return True

    def _check_end_screens_and_cards(self, video_details: List[Dict[str, Any]]) -> bool:
        # This is a placeholder. In a real implementation, you'd need to analyze the video content or use a separate API call to check for end screens and cards.
        return True

    async def competitor_analysis(self, channel_id: str) -> Dict[str, Any]:
        channel_data = await self._get_channel_data(channel_id)
        video_ids = await self._get_channel_video_ids(channel_id)
        video_details = await self._get_video_details(video_ids[:50])  # Analyze up to 50 most recent videos
        
        competitor_channels = await self._find_competitor_channels(channel_data['snippet']['title'])
        competitor_data = await asyncio.gather(*[self.analyze_channel(comp_id) for comp_id in competitor_channels])
        
        analysis = {
            "channel_stats": await self.analyze_channel(channel_id),
            "competitor_comparison": self._compare_with_competitors(channel_data, video_details, competitor_data),
            "content_gap_analysis": self._analyze_content_gaps(video_details, competitor_data),
            "keyword_opportunities": self._find_keyword_opportunities(video_details, competitor_data),
            "collaboration_opportunities": self._identify_collaboration_opportunities(competitor_channels)
        }
        
        return analysis

    async def _find_competitor_channels(self, channel_name: str) -> List[str]:
        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "q": channel_name,
            "type": "channel",
            "maxResults": 10,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return [item['snippet']['channelId'] for item in data['items'] if item['snippet']['channelTitle'] != channel_name]

    def _compare_with_competitors(self, channel_data: Dict[str, Any], video_details: List[Dict[str, Any]], competitor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        channel_stats = {
            "subscriber_count": int(channel_data['statistics']['subscriberCount']),
            "view_count": int(channel_data['statistics']['viewCount']),
            "video_count": int(channel_data['statistics']['videoCount']),
            "avg_views_per_video": self._calculate_avg_views(video_details),
            "engagement_rate": self._calculate_channel_engagement_rate(video_details)
        }
        
        competitor_stats = {
            "subscriber_count": [],
            "view_count": [],
            "video_count": [],
            "avg_views_per_video": [],
            "engagement_rate": []
        }
        
        for competitor in competitor_data:
            competitor_stats["subscriber_count"].append(competitor["subscriber_count"])
            competitor_stats["view_count"].append(competitor["view_count"])
            competitor_stats["video_count"].append(competitor["video_count"])
            competitor_stats["avg_views_per_video"].append(competitor["avg_views_per_video"])
            competitor_stats["engagement_rate"].append(competitor["engagement_rate"])
        
        comparison = {}
        for metric in channel_stats:
            channel_value = channel_stats[metric]
            competitor_values = competitor_stats[metric]
            avg_competitor_value = sum(competitor_values) / len(competitor_values)
            percentile = sum(1 for value in competitor_values if value < channel_value) / len(competitor_values) * 100
            comparison[metric] = {
                "channel_value": channel_value,
                "avg_competitor_value": avg_competitor_value,
                "percentile": percentile
            }
        
        return comparison

    def _analyze_content_gaps(self, video_details: List[Dict[str, Any]], competitor_data: List[Dict[str, Any]]) -> List[str]:
        channel_topics = set(self._extract_topics(video_details))
        competitor_topics = set()
        for competitor in competitor_data:
            competitor_topics.update(self._extract_topics(competitor["top_performing_videos"]))
        
        return list(competitor_topics - channel_topics)

    def _extract_topics(self, videos: List[Dict[str, Any]]) -> List[str]:
        topics = []
        for video in videos:
            text = video['title'] if 'title' in video else video['snippet']['title']
            topics.extend(self._extract_keywords([text]))
        return topics

    def _find_keyword_opportunities(self, video_details: List[Dict[str, Any]], competitor_data: List[Dict[str, Any]]) -> List[str]:
        channel_keywords = set(self._analyze_channel_keywords(video_details).keys())
        competitor_keywords = set()
        for competitor in competitor_data:
            competitor_keywords.update(competitor["keyword_analysis"].keys())
        
        return list(competitor_keywords - channel_keywords)

    def _identify_collaboration_opportunities(self, competitor_channels: List[str]) -> List[Dict[str, Any]]:
        # This is a placeholder. In a real implementation, you'd analyze the competitor channels to find good collaboration fits.
        return [{"channel_id": channel_id, "collaboration_score": 0.5} for channel_id in competitor_channels]

    async def trend_analysis(self) -> Dict[str, Any]:
        trending_videos = await self._get_trending_videos()
        
        analysis = {
            "top_trending_topics": self._extract_trending_topics(trending_videos),
            "trending_video_stats": self._analyze_trending_video_stats(trending_videos),
            "trending_tags": self._extract_trending_tags(trending_videos),
            "trending_titles": self._analyze_trending_titles(trending_videos),
            "trending_thumbnails": self._analyze_trending_thumbnails(trending_videos)
        }
        
        return analysis

    async def _get_trending_videos(self) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/videos"
        params = {
            "part": "snippet,statistics",
            "chart": "mostPopular",
            "regionCode": "US",
            "maxResults": 50,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['items']

    async def _extract_trending_topics(self, trending_videos: List[Dict[str, Any]]) -> List[str]:
        all_titles = [video['snippet']['title'] for video in trending_videos]
        return self._extract_keywords(all_titles)

    async def _analyze_trending_video_stats(self, trending_videos: List[Dict[str, Any]]) -> Dict[str, float]:
        view_counts = [int(video['statistics']['viewCount']) for video in trending_videos]
        like_counts = [int(video['statistics'].get('likeCount', 0)) for video in trending_videos]
        comment_counts = [int(video['statistics'].get('commentCount', 0)) for video in trending_videos]
        
        return {
            "avg_views": sum(view_counts) / len(view_counts),
            "avg_likes": sum(like_counts) / len(like_counts),
            "avg_comments": sum(comment_counts) / len(comment_counts),
            "avg_engagement_rate": sum((likes + comments) / views for views, likes, comments in zip(view_counts, like_counts, comment_counts)) / len(trending_videos)
        }

    async def _extract_trending_tags(self, trending_videos: List[Dict[str, Any]]) -> List[str]:
        all_tags = []
        for video in trending_videos:
            all_tags.extend(video['snippet'].get('tags', []))
        return [tag for tag, _ in Counter(all_tags).most_common(20)]

    async def _analyze_trending_titles(self, trending_videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        titles = [video['snippet']['title'] for video in trending_videos]
        title_lengths = [len(title) for title in titles]
        words_in_titles = [word for title in titles for word in title.split()]
        
        return {
            "avg_title_length": sum(title_lengths) / len(title_lengths),
            "common_title_words": [word for word, _ in Counter(words_in_titles).most_common(10) if word.lower() not in self.stop_words],
            "title_sentiment": self._analyze_sentiment(trending_videos)
        }

    async def _analyze_trending_thumbnails(self, trending_videos: List[Dict[str, Any]]) -> Dict[str, Any]:
        # This is a placeholder. In a real implementation, you'd use image analysis to extract color schemes, text usage, etc.
        return {
            "avg_text_percentage": 30,
            "common_colors": ["red", "blue", "yellow"],
            "face_percentage": 60
        }

    async def real_time_monitoring(self, video_id: str, duration: int = 60) -> Dict[str, List[Dict[str, Any]]]:
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration)
        
        metrics = {
            "views": [],
            "likes": [],
            "comments": []
        }
        
        while datetime.now() < end_time:
            video_stats = await self._get_video_stats(video_id)
            current_time = datetime.now()
            
            metrics["views"].append({"time": current_time, "value": int(video_stats["viewCount"])})
            metrics["likes"].append({"time": current_time, "value": int(video_stats.get("likeCount", 0))})
            metrics["comments"].append({"time": current_time, "value": int(video_stats.get("commentCount", 0))})
            
            await asyncio.sleep(60)  # Wait for 1 minute before the next data point
        
        return metrics

    async def _get_video_stats(self, video_id: str) -> Dict[str, str]:
        url = f"{self.base_url}/videos"
        params = {
            "part": "statistics",
            "id": video_id,
            "key": self.api_key
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data['items'][0]['statistics']

    async def run_full_analysis(self, keyword: str, video_id: str, channel_id: str) -> Dict[str, Any]:
        tasks = [
            self.analyze_keyword(keyword),
            self.analyze_video(video_id, keyword),
            self.analyze_channel(channel_id),
            self.competitor_analysis(channel_id),
            self.trend_analysis(),
            self.real_time_monitoring(video_id, duration=5)  # Run for 5 minutes as an example
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            "keyword_analysis": results[0],
            "video_analysis": results[1],
            "channel_analysis": results[2],
            "competitor_analysis": results[3],
            "trend_analysis": results[4],
            "real_time_monitoring": results[5]
        }
"""
# Example usage
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable.")

seo_tool = YouTubeSEOTool(api_key)

keyword = input("Enter keyword: ")  # Move the input here
video_id = "_gGo_AESIwk"
channel_id = "UCbT1KCssk84AmC7X7Ruvl0g"

full_analysis = await seo_tool.run_full_analysis(keyword=keyword, video_id=video_id, channel_id=channel_id)
print(json.dumps(full_analysis, indent=2))
"""

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