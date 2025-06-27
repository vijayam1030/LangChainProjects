@echo off
REM Windows batch script for Docker deployment

setlocal enabledelayedexpansion

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    exit /b 1
)

REM Check for docker-compose
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] docker-compose not found, trying docker compose
    set DOCKER_COMPOSE=docker compose
) else (
    set DOCKER_COMPOSE=docker-compose
)

if "%1"=="build" (
    echo [INFO] Building Docker images...
    docker build -t langchain-apps:latest .
    docker build -f Dockerfile.prod -t langchain-apps:prod .
    echo [INFO] Images built successfully!
    goto :eof
)

if "%1"=="start" (
    if "%2"=="" (
        echo [INFO] Starting all LangChain services...
        %DOCKER_COMPOSE% up -d
        echo [INFO] All services started!
        echo [INFO] Services available at:
        echo   - Multi-model app: http://localhost:7860
        echo   - RAG Streamlit:   http://localhost:8501
        echo   - RAG Gradio:      http://localhost:7861
        echo   - MCP Server:      http://localhost:8000
        echo   - Ollama:          http://localhost:11434
    ) else (
        echo [INFO] Starting %2 service...
        %DOCKER_COMPOSE% up -d %2
        echo [INFO] %2 service started!
    )
    goto :eof
)

if "%1"=="stop" (
    echo [INFO] Stopping all services...
    %DOCKER_COMPOSE% down
    echo [INFO] All services stopped!
    goto :eof
)

if "%1"=="restart" (
    echo [INFO] Restarting all services...
    %DOCKER_COMPOSE% down
    %DOCKER_COMPOSE% up -d
    echo [INFO] All services restarted!
    goto :eof
)

if "%1"=="logs" (
    if "%2"=="" (
        %DOCKER_COMPOSE% logs -f
    ) else (
        %DOCKER_COMPOSE% logs -f %2
    )
    goto :eof
)

if "%1"=="status" (
    echo [INFO] Service status:
    %DOCKER_COMPOSE% ps
    goto :eof
)

if "%1"=="cleanup" (
    echo [INFO] Cleaning up Docker resources...
    %DOCKER_COMPOSE% down -v
    docker system prune -f
    echo [INFO] Cleanup complete!
    goto :eof
)

REM Default help message
echo Usage: %0 {build^|start [service]^|stop^|restart^|logs [service]^|status^|cleanup}
echo.
echo Commands:
echo   build           - Build Docker images
echo   start [service] - Start all services or specific service
echo   stop            - Stop all services
echo   restart         - Restart all services
echo   logs [service]  - View logs for all or specific service
echo   status          - Show service status
echo   cleanup         - Stop services and clean up resources
echo.
echo Available services: multimodel, rag-streamlit, rag-gradio, mcp-server, ollama
