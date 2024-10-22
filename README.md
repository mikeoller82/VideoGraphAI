<div align="center">

# VideoGraphAI 🎬

![VideoGraphAI](https://img.shields.io/badge/VideoGraphAI-v1.0-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.0%2B-red.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/mikeoller82/VideoGraphAI/issues)
![Status](https://img.shields.io/badge/Status-Beta-yellow.svg)

An open-source AI-powered YouTube Shorts automation tool that revolutionizes content creation using graph-based agents and state-of-the-art AI technologies.

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Contributing](#-contributing) • [License](#-license)

</div>

## 📚 Table of Contents

- [🌟 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🔄 Workflow](#-workflow)
- [📋 Prerequisites](#-prerequisites)
- [🚀 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [📝 Usage](#-usage)
- [🔧 Troubleshooting](#-troubleshooting)
- [👥 Contributing](#-contributing)
- [🙏 Acknowledgements](#-acknowledgements)
- [📄 License](#-license)

## 🌟 Overview

VideoGraphAI streamlines the creation of YouTube Shorts using advanced AI technologies. Built with Streamlit, it offers end-to-end video production capabilities from content research to final rendering. The system leverages various AI models and APIs to create engaging, relevant content automatically.

## ✨ Key Features

- 🔍 **Real-time Research**: Automated content research using Tavily Search API
- 📝 **AI Script Generation**: Flexible LLM compatibility (OpenAI, Groq, etc.)
- 🎨 **Dynamic Visuals**: Image generation via TogetherAI (FLUX.schnell)
- 🎤 **Professional Audio**: Voiceovers using F5-TTS
- 📺 **Automated Subtitles**: Synchronized captions with Gentle
- 🖥️ **User-Friendly Interface**: Built with Streamlit for easy operation

## 🔄 Workflow

1. **Input** → User provides topic, timeframe, and video length
2. **Research** → AI researches recent events using graph agents
3. **Content Creation** → Generates titles, descriptions, hashtags, and script
4. **Media Production** → Creates storyboard and acquires media assets
5. **Audio & Subtitles** → Generates voiceover and synchronized captions
6. **Compilation** → Assembles final video with all components
7. **Delivery** → Presents downloadable video through Streamlit interface

## 📋 Prerequisites

- Python 3.8+
- FFmpeg
- Docker (optional, recommended for Gentle server)
- API Keys:
  - Groq API
  - Together AI API
  - Tavily Search API
  - F5-TTS (local installation)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/mikeoller82/videographai.git
cd videographai
```

### 2. Environment Setup
```bash
# Option 1: Conda (Recommended)
conda create -n videographai python=3.8 pip
conda activate videographai

# Option 2: Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. FFmpeg Installation

<details>
<summary>Click to expand installation instructions for your OS</summary>

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
- Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add bin folder to system PATH
</details>

### 5. F5-TTS Setup
```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install -r requirements.txt
# Follow F5-TTS documentation for torch and CUDA setup
cd ..
```

## ⚙️ Configuration

Create a `.env` file:
```env
GROQ_API_KEY=your_groq_api_key
BFL_API_KEY=your_black_forest_labs_api_key
TOGETHER_API_KEY=your_together_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 📝 Usage

1. Launch application:
```bash
streamlit run app.py
```

2. Enter parameters:
   - Topic for your video
   - Time frame (past month/year/all)
   - Video length (60/120/180 seconds)

3. Click "Generate Video" and wait for processing

## 🔧 Troubleshooting

<details>
<summary>Common Issues and Solutions</summary>

- **API Issues**: Verify API keys in `.env`
- **Gentle Server**: Ensure server is running on port 8765
- **FFmpeg**: Confirm PATH configuration
- **Dependencies**: Check virtual environment activation
- **Video Issues**: Review application logs
- **UI Problems**: Clear browser cache
</details>

## 👥 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgements

<div align="center">

### Powered By

[<img src="https://groq.com/wp-content/uploads/2024/03/PBG-mark1-color.svg" width="200" alt="Groq">](https://groq.com)

</div>

- **F5-TTS**: Advanced text-to-speech capabilities
```bibtex
@article{chen-etal-2024-f5tts,
    title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
    author={Chen, Yushen and Niu, Zhikang and Ma, Ziyang and Deng, Keqi and Wang, Chunhui and Zhao, Jian and Yu, Kai and Chen, Xie},
    journal={arXiv preprint arXiv:2410.06885},
    year={2024}
}
```
- **TogetherAI**: Image generation via FLUX.schnell model
  (https://www.together.ai/)
  
- **Tavily**: Real-time search capabilities
  (https://tavily.com/#api)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">

Made with ❤️ by the VideoGraphAI Community

[⭐ Star us on GitHub](https://github.com/mikeoller82/videographai)

</div>
