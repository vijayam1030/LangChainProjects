#!/bin/bash

# Docker deployment scripts for LangChain applications

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose not found, trying docker compose"
        DOCKER_COMPOSE="docker compose"
    else
        DOCKER_COMPOSE="docker-compose"
    fi
}

# Build all images
build_images() {
    print_status "Building Docker images..."
    docker build -t langchain-apps:latest .
    docker build -f Dockerfile.prod -t langchain-apps:prod .
    print_status "Images built successfully!"
}

# Start all services
start_all() {
    print_status "Starting all LangChain services..."
    $DOCKER_COMPOSE up -d
    print_status "All services started!"
    print_status "Services available at:"
    echo "  - Multi-model app: http://localhost:7860"
    echo "  - RAG Streamlit:   http://localhost:8501"
    echo "  - RAG Gradio:      http://localhost:7861"
    echo "  - MCP Server:      http://localhost:8000"
    echo "  - Ollama:          http://localhost:11434"
}

# Start specific service
start_service() {
    if [ -z "$1" ]; then
        print_error "Please specify a service name"
        echo "Available services: multimodel, rag-streamlit, rag-gradio, mcp-server, ollama"
        exit 1
    fi
    
    print_status "Starting $1 service..."
    $DOCKER_COMPOSE up -d $1
    print_status "$1 service started!"
}

# Stop all services
stop_all() {
    print_status "Stopping all services..."
    $DOCKER_COMPOSE down
    print_status "All services stopped!"
}

# View logs
view_logs() {
    if [ -z "$1" ]; then
        $DOCKER_COMPOSE logs -f
    else
        $DOCKER_COMPOSE logs -f $1
    fi
}

# Show status
show_status() {
    print_status "Service status:"
    $DOCKER_COMPOSE ps
}

# Clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    $DOCKER_COMPOSE down -v
    docker system prune -f
    print_status "Cleanup complete!"
}

# Main script logic
case "$1" in
    "build")
        check_docker
        build_images
        ;;
    "start")
        check_docker
        if [ -z "$2" ]; then
            start_all
        else
            start_service $2
        fi
        ;;
    "stop")
        check_docker
        stop_all
        ;;
    "restart")
        check_docker
        stop_all
        start_all
        ;;
    "logs")
        check_docker
        view_logs $2
        ;;
    "status")
        check_docker
        show_status
        ;;
    "cleanup")
        check_docker
        cleanup
        ;;
    *)
        echo "Usage: $0 {build|start [service]|stop|restart|logs [service]|status|cleanup}"
        echo ""
        echo "Commands:"
        echo "  build           - Build Docker images"
        echo "  start [service] - Start all services or specific service"
        echo "  stop            - Stop all services"
        echo "  restart         - Restart all services"
        echo "  logs [service]  - View logs for all or specific service"
        echo "  status          - Show service status"
        echo "  cleanup         - Stop services and clean up resources"
        echo ""
        echo "Available services: multimodel, rag-streamlit, rag-gradio, mcp-server, ollama"
        exit 1
        ;;
esac
