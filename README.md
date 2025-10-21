# personal-financial-ai-agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Latest Release](https://img.shields.io/github/v/release/merendamattia/personal-financial-ai-agent?label=release)](https://github.com/merendamattia/personal-financial-ai-agent/releases)

A personal financial advisor AI agent: get intelligent financial guidance in your preferred language, with local LLM inference!

## Requirements

- Python 3.11+
- Ollama installed and running

## Quick Installation

1. Install Ollama: download and install from [ollama.com](https://ollama.com/) for your operating system.

2. Start Ollama and pull a model

3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

4. Install dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Prepare environment variables:

```bash
cp .env.example .env
```

Edit `.env` to set required API keys and/or configure Ollama.

## Usage

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser and start chatting.


## Contributing

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. Please install the Git commit hooks before making commits:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```
