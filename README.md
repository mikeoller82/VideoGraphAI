# YouTube Shorts Generator

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Async](https://img.shields.io/badge/async-IO-brightgreen.svg)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Workflow](#workflow)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Example](#example)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

YouTube Shorts Generator is an automated script designed to streamline the creation of YouTube Shorts. Leveraging asynchronous operations and various AI agents, this tool handles everything from researching recent events to compiling the final video with voiceovers and subtitles. Whether you're a content creator looking to scale your production or an enthusiast interested in automated video generation, this tool provides a comprehensive solution.

## Features

- **Automated Research**: Gathers recent events related to your chosen topic.
- **AI-Powered Content Generation**: Utilizes multiple AI agents for crafting titles, descriptions, hashtags, scripts, and storyboards.
- **Media Integration**: Searches and integrates relevant videos and images from Pexels and Pixabay.
- **Voiceover and Subtitles**: Generates synchronized voiceovers and subtitles using Gentle and TikTokVoice.
- **Video Compilation**: Assembles video clips, images, voiceovers, and subtitles into a polished YouTube Short using FFmpeg.
- **Error Handling and Logging**: Comprehensive logging for monitoring and debugging.

## Workflow

1. **Research**: The script starts by researching recent events based on the provided topic and timeframe.
2. **Content Generation**:
   - Generates multiple title suggestions and selects the most effective one.
   - Creates an optimized video description.
   - Generates relevant hashtags and tags.
   - Develops a concise video script tailored for YouTube Shorts.
3. **Storyboard Creation**: Translates the script into a detailed storyboard with visual and textual elements.
4. **Media Acquisition**: Downloads necessary video clips and images from stock sources.
5. **Voiceover and Subtitles**: Produces a voiceover for the script and generates synchronized subtitles.
6. **Video Compilation**: Compiles all elements into a final YouTube Short video.

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Ensure FFmpeg is installed and accessible in your system's PATH.
- **Docker** (optional but recommended for running the Gentle server)
- **Gentle Server**: For audio-text alignment.

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mikeoller82/youtube-shorts-generator.git
   cd youtube-shorts-generator
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install and Configure FFmpeg**

   - **Ubuntu/Debian**

     ```bash
     sudo apt update
     sudo apt install ffmpeg
     ```

   - **macOS (using Homebrew)**

     ```bash
     brew install ffmpeg
     ```

   - **Windows**

     - Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
     - Extract and add the `bin` folder to your system's PATH.

## Configuration

1. **Environment Variables**

   Create a `.env` file in the project root directory and add the following API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   PEXELS_API_KEY=your_pexels_api_key
   TAVILY_API_KEY=your_tavily_api_key
   PIXABAY_API_KEY=your_pixabay_api_key
   TIKTOK_SESSION_ID=your_tiktok_session_id
   ```

   **Note**: Replace `your_*` with your actual API keys. Ensure that the `.env` file is kept secure and is not committed to version control.

2. **Gentle Server Setup**

   The script uses Gentle for audio-text alignment. You can set it up using Docker:

   ```bash
   docker run -d -p 8765:8765 lowerquality/gentle
   ```

   **Alternatively**, you can install Gentle locally following the instructions on [Gentle's GitHub](https://github.com/lowerquality/gentle).

## Usage

Run the script using Python:

```bash
python youtube_shorts_generator.py
```

**Upon Execution**, you will be prompted to provide:

1. **Topic**: The subject for your YouTube Short.
2. **Time Frame**: The period for recent events (e.g., 'past week', 'past year', 'all').
3. **Video Length**: Desired length of the video in seconds (e.g., 60).

The script will execute the workflow and generate a YouTube Short based on your inputs.

## Example

1. **Run the Script**

   ```bash
   python youtube_shorts_generator.py
   ```

2. **Provide Inputs**

   ```
   Enter the topic for your YouTube video: Unexplained Mysteries
   Enter the time frame for recent events (e.g., 'past week', '30d', '1y'): past month
   Enter the desired video length in seconds (e.g., 60): 60
   ```

3. **Output**

   The script will process the inputs, generate necessary content, and compile the final video. Upon successful execution, you will see:

   ```
   YouTube Short saved as 'youtube_short.mp4'
   ```

   The video will be located in the current working directory.

## Troubleshooting

- **Missing API Keys**: Ensure all required API keys are correctly set in the `.env` file.
- **Gentle Server Errors**: Verify that the Gentle server is running and accessible at `http://localhost:8765`.
- **FFmpeg Issues**: Ensure FFmpeg is installed and correctly added to your system's PATH.
- **Dependency Issues**: Make sure all Python dependencies are installed without errors. Consider using a virtual environment.
- **Video Compilation Failures**: Check the logs for specific errors. Ensure that all media files are correctly downloaded and accessible.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add YourFeature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).
