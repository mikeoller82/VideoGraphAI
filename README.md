## VideoGraphAI 🎬

VideoGraphAI is an open-source automation tool that revolutionizes YouTube Shorts creation using AI and graph-based agents. Built with Streamlit, it offers end-to-end video production capabilities from content research to final rendering.

## 🚀 Key Features:
- Real-time content research using Tavily Search API
- AI-powered script generation with LLM compatibility (OpenAI, Groq, etc.)
- Dynamic image generation via TogetherAI (FLUX.schnell)
- Professional voiceovers using F5-TTS
- Automated subtitle generation with Gentle
- Streamlit-based user interface for easy operation

Perfect for content creators, marketers, and AI enthusiasts looking to automate short-form video production. Currently in beta - contributors welcome! 

🤝 Join us in developing the future of AI-driven content creation.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-red.svg)


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

VideoGraphAI is a Streamlit-based application designed to streamline the creation of YouTube Shorts. Leveraging graph agents with Groq's API Endpoints ( you can honestly change it to whatevr endpoint you want with OpenAI Compatibilty  would work all the sam maybe better) technologies, this tool handles everything from researching recent events with realtime web search capabilties thru Tavily Sarch API to compiling the final video. Uses TogetherAI API for FLUX.schnell images, voiceovers from F5-TTS and subtitles with Gentle. Whether you're a content creator looking to scale your production or an enthusiast interested in automated video generation, VideoGraphAI provides a starting point as its very beta, I beg for contributions and  solutions.#VideoAI #ContentCreation #YouTubeAutomation #AITools #OpenSourceAI #StreamlitApp #PythonDev #ContentCreator #AIVideo #AutomatedContent

## Workflow

1. **User Input**: Through the Streamlit interface, users provide the topic, timeframe, and desired video length.
2. **Research**: The application researches recent events based on the provided topic and timeframe using graph agents.
3. **Content Generation**:
   - Generates multiple title suggestions and selects the most effective one.
   - Creates an optimized video description.
   - Generates relevant hashtags and tags.
   - Develops a concise video script tailored for YouTube Shorts.
4. **Storyboard Creation**: Translates the script into a detailed storyboard with visual and textual elements.
5. **Media Acquisition**: Downloads necessary video clips and images from stock sources.
6. **Voiceover and Subtitles**: Produces a voiceover for the script and generates synchronized subtitles.
7. **Video Compilation**: Compiles all elements into a final YouTube Short video.
8. **Result Presentation**: Displays the generated video and provides download options through the Streamlit interface.

## Prerequisites

- **Python 3.8+**
- **FFmpeg**: Ensure FFmpeg is installed and accessible in your system's PATH.
- **Docker** (optional but recommended for running the Gentle server)
- **Gentle Server**: For audio-text alignment.
- **F5-TTS** : (Voiceover)
- **GROQ API** (LLM's)
- **TOGETHER AI API** (images for scenes)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/mikeoller82/videographai.git
   cd videographai
   ```

2. **Create a Virtual Environment**(I recommend Conda but you can figure out how to do that yourself)
    ```bash
    conda create -n videographai python=3.8 pip
    conda activate videographai
     ```

     or

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Install and Configure FFmpeg**

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
    
6. ** You have to git clone inside your working directory the F5-TTS model which is open source , read threw their docs, super straightforward imo you shouldnt have issues**
   ```bash
   git clone https://github.com/SWivid/F5-TTS.git
   cd F5-TTS
   pip install -r requirementts.txt
   
   # Follow their docs for other steps which iis install torch and cuda no biggie
   cd ..
   ```

## Configuration

1. **Environment Variables**

   Create a `.env` file in the project root directory and add the following API keys:

   ```env
   GROQ_API_KEY=your_groq_api_key
   BFL_API_KEY=your_black_forest_labs_api_key
   TOGETHER_API_KEY=your_together_api_key
   TAVILY_API_KEY=your_tavily_api_key
   
   #TIKTOK_SESSION_ID=your_tiktok_session_id
   #(get yout tik tok session id byt going to tiktok, right click anywhere , go to inspect source , then click on arrows that extends menu at top , click applications, search for session id..super simple)
   # TTS has been updated to use F5-TTS so the tiktoksession is no longer needed
   ```

   **Note**: Replace `your_*` with your actual API keys. Ensure that the `.env` file is kept secure and is not committed to version control.

2. **Gentle Server Setup**

   The application uses Gentle for audio-text alignment. You can set it up using Docker:

   ```bash
   docker run -d -p 8765:8765 lowerquality/gentle
   ```

   **Alternatively**, you can install Gentle locally following the instructions on [Gentle's GitHub](https://github.com/lowerquality/gentle).

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

This will launch the VideoGraphAI web interface in your default browser. Follow the on-screen instructions to:

1. Enter your desired topic for the YouTube Short.
2. Specify the time frame for recent events.
3. Set the desired video length.
4. Click the "Generate Video" button to start the process.

The application will display progress updates and eventually present the final video for download.

## Example

1. **Launch the Application**

   ```bash
   streamlit run app.py
   ```

2. **Input Parameters**

   - Topic: "whatever topic you want a vidoe on"
   - Time Frame: "past month, past year, all" all gives you all time search
   - Video Length: 60, 120, 180....it goes by seconds just keep that in mnd or it defualts to 60

3. **Generate Video**

   Click the "Generate Video" button and wait for the process to complete.

4. **Result**

   The application will display the generated video and it will download toyour working directory to youtube_short.mp4 i believe i forget sorry.

## Troubleshooting

- **API Key Issues**: Ensure all required API keys are correctly set in the `.env` file.
- **Gentle Server Errors**: Verify that the Gentle server is running and accessible at `http://localhost:8765`.
- **FFmpeg Issues**: Ensure FFmpeg is installed and correctly added to your system's PATH.
- **Dependency Issues**: Make sure all Python dependencies are installed without errors. Consider using a virtual environment.
- **Video Compilation Failures**: Check the application logs for specific errors. Ensure that all media files are correctly downloaded and accessible.
- **Streamlit Interface Problems**: Clear your browser cache or try a different browser if the interface is not loading correctly.

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


## Acknowledgements

**1. GROQ (LLM's for Agents)**

<a href="https://groq.com" target="_blank" rel="noopener noreferrer">
  <img
    src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg"
    alt="Powered by Groq for fast inference."
  />
</a>

**2. F5-TTS (Narration Voiceovers)**

@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}

**3.TogetherAI (Free Image genrations for FLUX.schnell model that creates images for scenes)**
https://github.com/togethercomputer



## License

This project is licensed under the [MIT License](LICENSE).
