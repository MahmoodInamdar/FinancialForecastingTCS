# TCS Financial Forecasting Agent

AI-powered financial analysis agent using LangGraph, CrewAI, and modern FastAPI streaming.

## Features

🚀 **Real-time Streaming**: WebSocket-based live updates
📊 **Smart Extraction**: Auto-validates PDF/Excel data extraction
🤖 **Multi-Agent**: CrewAI + LangGraph orchestration
📈 **Financial Analysis**: Revenue, margins, growth trends
💾 **MySQL Logging**: All forecasts logged automatically
🎯 **Interactive UI**: Modern streaming interface

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pymupdf pandas mysql-connector-python
pip install langchain langgraph crewai

# Run the app
python app.py

# Open browser
open frontend.html
```

## API Endpoints

**POST /forecast** - Generate forecast
**WebSocket /ws/forecast** - Streaming analysis
**GET /documents** - List available documents
**POST /upload** - Upload new documents

## Architecture

```
Frontend (HTML/JS)
    ↓ WebSocket/HTTP
FastAPI App
    ↓ LangGraph Workflow
[Document Loader] → [Financial Extractor] → [Qualitative Analyzer] → [Forecaster]
    ↓ CrewAI Validation
[Financial Analyst] + [Market Analyst] + [Forecaster]
    ↓ MySQL Logging
Database Storage
```

## Tools

- **FinancialDataExtractorTool**: Validates PDF/Excel extraction
- **QualitativeAnalysisTool**: Analyzes transcripts with confidence scoring

## Data

Place TCS financial documents in `data/` folder:
- PDFs: Quarterly reports, transcripts
- Excel: Financial data sheets

The system auto-validates extraction quality and provides confidence scores.

## 2025 Tech Stack

- **FastAPI**: Production async web framework
- **LangGraph**: State-driven AI workflows
- **CrewAI**: Multi-agent collaboration
- **WebSockets**: Real-time streaming
- **MySQL**: Request logging
- **PyMuPDF**: High-performance PDF processing