# main.py
import asyncio
import yaml
from plugins.agents import check_api_keys
from loguru import logger
from flask import Flask, render_template, request, jsonify
from plugins.plugin_loader import load_plugins
from plugins.state import youtube_optimization_workflow
from database.db_manager import DatabaseManager

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

value = config.get('database', {}).get('path')
print(value)


# Set up logging
logger.add("app.log", rotation="500 MB")

# Load plugins
agents, tools = load_plugins()

# Initialize database
db_manager = DatabaseManager(config['database']['path'])

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
        results = await youtube_optimization_workflow(topic, duration_minutes, time_frame, agents, tools, db_manager)
        
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

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
async def optimize():
    topic = request.form['topic']
    duration_minutes = int(request.form['duration'])
    time_frame = request.form['time_frame']

    try:
        results = await youtube_optimization_workflow(topic, duration_minutes, time_frame, agents, tools, db_manager)
        return jsonify(results)
    except Exception as e:
        logger.exception(f"Error in optimization process: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    asyncio.run(main())
