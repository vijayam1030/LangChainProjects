# Docker Setup for LangChain Applications

This guide explains how to containerize and deploy all your LangChain applications using Docker.

## 🐳 Docker Files Overview

- **`Dockerfile`** - Main development image
- **`Dockerfile.prod`** - Production-ready image with security hardening
- **`docker-compose.yml`** - Multi-service orchestration
- **`.dockerignore`** - Excludes unnecessary files from build
- **`.env.docker`** - Environment variables template
- **`deploy.sh/.bat`** - Deployment automation scripts

## 🚀 Quick Start

### Option 1: Single Service (Quick Test)
```bash
# Build and run multimodel app only
docker build -t langchain-multimodel .
docker run -p 7860:7860 -e OLLAMA_BASE_URL=http://host.docker.internal:11434 langchain-multimodel
```

### Option 2: All Services (Recommended)
```bash
# On Linux/Mac
./deploy.sh build
./deploy.sh start

# On Windows
deploy.bat build
deploy.bat start
```

## 📋 Available Services

| Service | Port | Description | URL |
|---------|------|-------------|-----|
| **multimodel** | 7860 | Multi-model comparison | http://localhost:7860 |
| **rag-streamlit** | 8501 | RAG Streamlit app | http://localhost:8501 |
| **rag-gradio** | 7861 | RAG Gradio app | http://localhost:7861 |
| **mcp-server** | 8000 | MCP server | http://localhost:8000 |
| **ollama** | 11434 | Ollama backend | http://localhost:11434 |

## 🛠️ Management Commands

### Using deploy scripts:
```bash
# Build images
./deploy.sh build

# Start all services
./deploy.sh start

# Start specific service
./deploy.sh start multimodel

# View logs
./deploy.sh logs
./deploy.sh logs multimodel

# Check status
./deploy.sh status

# Stop all services
./deploy.sh stop

# Cleanup everything
./deploy.sh cleanup
```

### Manual Docker commands:
```bash
# Build development image
docker build -t langchain-apps:latest .

# Build production image
docker build -f Dockerfile.prod -t langchain-apps:prod .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## ⚙️ Configuration

### Environment Variables (.env.docker)
```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
GRADIO_SERVER_NAME=0.0.0.0
MAX_WORKERS=4
WIKIPEDIA_MAX_DOCS=5
```

### Custom Configuration
1. Copy `.env.docker` to `.env`
2. Modify values as needed
3. Restart services

## 🔧 Troubleshooting

### Common Issues:

**1. Ollama Connection Refused**
```bash
# Make sure Ollama is running on host
ollama serve

# Or run Ollama in Docker too
docker-compose up -d ollama
```

**2. Port Already in Use**
```bash
# Check what's using the port
netstat -tulpn | grep :7860

# Stop the service or change port in docker-compose.yml
```

**3. Build Fails**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t langchain-apps .
```

**4. Permission Denied (Linux)**
```bash
# Make scripts executable
chmod +x deploy.sh

# Or run with bash
bash deploy.sh build
```

## 🏗️ Production Deployment

### Using Production Image:
```bash
# Build production image
docker build -f Dockerfile.prod -t langchain-apps:prod .

# Run with production settings
docker run -d \
  --name langchain-multimodel \
  -p 7860:7860 \
  -e OLLAMA_BASE_URL=http://your-ollama-server:11434 \
  --restart unless-stopped \
  langchain-apps:prod
```

### Using Docker Swarm:
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml langchain
```

### Health Checks:
```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"

# View health check logs
docker inspect --format='{{json .State.Health}}' container_name
```

## 📈 Performance Optimization

### Resource Limits:
```yaml
# In docker-compose.yml
services:
  multimodel:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
```

### Caching:
- Images use multi-stage builds for smaller size
- Python packages cached in Docker layers
- Wikipedia data cached in shared volumes

### Scaling:
```bash
# Scale specific service
docker-compose up -d --scale multimodel=3

# Use load balancer (nginx, traefik, etc.)
```

## 🔐 Security Best Practices

1. **Non-root user**: Production image runs as non-root
2. **Minimal base image**: Uses slim Python image
3. **No secrets in image**: Environment variables for config
4. **Health checks**: Automatic service monitoring
5. **Resource limits**: Prevent resource exhaustion

## 📊 Monitoring

### View Resource Usage:
```bash
# Container stats
docker stats

# Service logs
docker-compose logs -f --tail=100

# System events
docker events
```

### Log Management:
```bash
# Rotate logs
docker-compose logs --tail=1000 > app.log

# Use external log driver
# Add to docker-compose.yml:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 🚀 Next Steps

1. **Set up CI/CD** with GitHub Actions
2. **Add monitoring** with Prometheus/Grafana
3. **Implement load balancing** with nginx
4. **Use container registry** for image distribution
5. **Add backup strategy** for persistent data

Need help with any of these setups? Let me know!
