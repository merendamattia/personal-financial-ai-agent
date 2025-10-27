# personal-financial-ai-agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Latest Release](https://img.shields.io/github/v/release/merendamattia/personal-financial-ai-agent?label=release)](https://github.com/merendamattia/personal-financial-ai-agent/releases)

A personal financial advisor AI agent: get intelligent financial guidance in your preferred language, with local LLM inference!

## Requirements

- Python 3.11+
- Ollama installed and running (optional)

## Quick Installation

1. Create and activate a virtual environment:
```bash
conda create --name personal-financial-ai-agent-env python=3.11.13
conda activate personal-financial-ai-agent-env
```

2. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Prepare environment variables:

```bash
cp .env.example .env
```

Edit `.env` to set required API keys and/or configure Ollama.

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and start chatting.

On first launch, a modal dialog will appear asking you to select your preferred LLM provider. Choose from the available options based on what you have configured.

### Docker
You can use Docker to avoid local installations.

With Docker Compose (includes optional Ollama container):
```bash
docker compose up
docker compose down # shutdown
```

This will start:
- **financial-ai-agent** on http://localhost:8501
- **ollama** on http://localhost:11434 (automatically downloads `qwen2:0.6b` model on first start)

The first startup may take a few minutes while Ollama downloads the model.

Or with docker run (without Ollama):
```bash
docker build --no-cache -t financial-ai-agent:local .
docker run -p 8501:8501 --env-file .env financial-ai-agent:local
```

Access at http://localhost:8501

## Supported LLM Providers

The agent supports multiple LLM providers. Simply configure the necessary credentials and select your preferred provider from the UI:

### Ollama (Local Models)

- **No API key required** - fully offline
- **Setup**: Download and install [Ollama](https://ollama.com/), then start it on your system
- **Model**: Configure `OLLAMA_MODEL` in `.env` (default: `qwen3:0.6b`)
- **API URL**: `OLLAMA_API_URL` in `.env` (default: `http://localhost:11434/v1`)

### Google Generative AI (Gemini)

- Get an API key from [AI Studio](https://aistudio.google.com/app/apikey)
- Set `GOOGLE_API_KEY` in `.env`
- Configure `GOOGLE_MODEL` in `.env` (default: `gemini-2.0-flash`)

### OpenAI (GPT Models)

- Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- Set `OPENAI_API_KEY` in `.env`
- Configure `OPENAI_MODEL` in `.env` (default: `gpt-4o-mini`)

### Provider Selection

The app automatically detects which providers are available based on your `.env` configuration and displays only those options. Select your preferred provider when you start the app, or simply click the provider button in the chat interface to switch providers.

## Contributing

Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines, including how to set up pre-commit hooks and follow our commit conventions.
