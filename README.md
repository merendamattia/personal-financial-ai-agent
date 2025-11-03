# 💰 Personal Financial AI Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Latest Release](https://img.shields.io/github/v/release/merendamattia/personal-financial-ai-agent?label=release)](https://github.com/merendamattia/personal-financial-ai-agent/releases)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-personal--financial--ai--agent-blue?logo=docker)](https://hub.docker.com/repository/docker/merendamattia/personal-financial-ai-agent)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

An intelligent personal financial advisor powered by AI. Get expert financial guidance in your preferred language with support for multiple LLM providers, including local offline inference with Ollama.

## ✨ Features

- 🤖 **Multi-Provider AI Support** - Choose from Ollama (local), Google Gemini, or OpenAI
- 💬 **Interactive Conversations** - Natural language financial discussions
- 📊 **Portfolio Analysis** - AI-generated portfolio recommendations based on your profile
- 📈 **Historical Data** - 10-year historical returns for analyzed assets
- 🛡️ **Privacy First** - Full offline support with Ollama
- 🌐 **Multi-Language** - Communicate in your preferred language
- 📥 **Profile Management** - Load, save, and download financial profiles as JSON

## 📋 Requirements

- **Python 3.11+**
- **Ollama** (optional, for local LLM inference)

## 🚀 Getting Started

### Installation

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/merendamattia/personal-financial-ai-agent.git
   cd personal-financial-ai-agent
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   conda create --name personal-financial-ai-agent python=3.11.13
   conda activate personal-financial-ai-agent
   ```

3. **Install dependencies:**
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and preferences
   ```

5. **Extract the dataset:**
   ```bash
   cd dataset
   unzip ETFs.zip
   cd ..
   ```

### Running the Application

**Local Streamlit:**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Docker Compose (recommended):**
```bash
# Start (includes Ollama container)
docker compose up

# Stop
docker compose down
```
Access at http://localhost:8501

**Docker (without Ollama):**
```bash
# Option 1: Build locally
docker build --no-cache -t financial-ai-agent:local .
docker run -p 8501:8501 --env-file .env financial-ai-agent:local

# Option 2: Use pre-built image from Docker Hub
docker pull merendamattia/personal-financial-ai-agent:latest
docker run -p 8501:8501 --env-file .env merendamattia/personal-financial-ai-agent:latest
```

> **Note:** On first launch, you'll be prompted to select your preferred LLM provider.

## 🤖 Supported LLM Providers

Choose your preferred AI provider based on your needs:

### 🦙 Ollama (Recommended for Privacy)
- **Cost:** Free
- **Privacy:** 100% offline, no data sent to external servers
- **Setup:**
  1. Download and install [Ollama](https://ollama.com/)
  2. Start Ollama: `ollama serve`
- **Configuration:**
  - `OLLAMA_MODEL` in `.env` (default: `qwen3:0.6b`)
  - `OLLAMA_API_URL` in `.env` (default: `http://localhost:11434/v1`)

### 🌐 Google Generative AI (Gemini)
- **Cost:** Free tier available, then pay-as-you-go
- **Privacy:** Cloud-based processing
- **Setup:**
  1. Get API key from [AI Studio](https://aistudio.google.com/app/apikey)
  2. Set `GOOGLE_API_KEY` in `.env`
- **Configuration:**
  - `GOOGLE_MODEL` in `.env` (default: `gemini-2.5-flash`)

### ✨ OpenAI (GPT Models)
- **Cost:** Pay-as-you-go, no free tier
- **Privacy:** Cloud-based processing
- **Setup:**
  1. Create account and get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
  2. Set `OPENAI_API_KEY` in `.env`
- **Configuration:**
  - `OPENAI_MODEL` in `.env` (default: `gpt-4.1-mini`)

**Provider Selection:**
The app detects available providers from your `.env` configuration. Select your preferred provider when starting the app or switch anytime using the sidebar button.

## 🧪 Testing

Run the test suite:
```bash
pytest -q # Quick run
pytest -v # Verbose output
```

See `tests/` directory for available test files.

## 🔧 How It Works

1. **Agent Selection** → Choose your preferred LLM provider
2. **Conversation** → Answer financial assessment questions
3. **Profile Extraction** → AI extracts your financial profile from responses
4. **Portfolio Generation** → RAG-enhanced advisor generates personalized portfolio
5. **Analysis** → View historical returns and investment recommendations

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for:
- Development setup
- Pre-commit hooks configuration
- Commit convention guidelines
- Pull request process

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## 📞 Support

For issues, feature requests, or questions:
- Open an [issue on GitHub](https://github.com/merendamattia/personal-financial-ai-agent/issues)
