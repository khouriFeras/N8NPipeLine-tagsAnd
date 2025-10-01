# Unified Product System

A comprehensive AI-powered system for e-commerce product management, combining automated tagging, translation, and SEO optimization in one unified platform.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Product Tagging](#product-tagging)
  - [Translation &amp; SEO](#translation--seo)
  - [Unified API](#unified-api)
- [API Reference](#api-reference)
- [n8n Integration](#n8n-integration)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## Overview

This unified system provides three major capabilities:

1.  **Automated Product Tagging**: AI-powered hierarchical taxonomy classification using semantic search
2. **Translation & SEO**: English-to-Arabic translation with SEO-optimized content generation
3. **n8n Automation**: Complete workflow automation with Google Sheets integration

Built with OpenAI GPT-4, FAISS vector search, and Flask REST APIs, this system is production-ready and designed for seamless integration with e-commerce platforms.

---

## Features

### Product Tagging System

- ✅ AI-powered taxonomy matching using semantic embeddings
- ✅ Hierarchical tag generation (L1 → L2 → L3 → L4)
- ✅ Bilingual descriptor support (English + Arabic)
- ✅ FAISS-based fast similarity search
- ✅ Confidence scoring and verification
- ✅ Batch processing with configurable workers

### Translation & SEO System

- ✅ English to Arabic product translation
- ✅ SEO-optimized meta titles (≤60 chars)
- ✅ SEO-optimized meta descriptions (≤155 chars)
- ✅ Arabic product titles (50-80 chars)
- ✅ SEO-friendly URL handles
- ✅ Async processing for high performance

### API & Integration

- ✅ RESTful API with Flask
- ✅ n8n workflow automation support
- ✅ Google Sheets integration
- ✅ Docker containerization
- ✅ CORS-enabled for frontend integration
- ✅ Health checks and status monitoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Product System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌─────────────────┐          │
│  │  Tagging Module  │         │ Translation API │          │
│  │                  │         │                 │          │
│  │ • OpenAI GPT-4   │         │ • OpenAI GPT-4  │          │
│  │ • FAISS Search   │         │ • SEO Generator │          │
│  │ • Taxonomy Tags  │         │ • Arabic Trans  │          │
│  └──────────────────┘         └─────────────────┘          │
│           │                            │                     │
│           └────────────┬───────────────┘                     │
│                        │                                     │
│              ┌─────────▼──────────┐                         │
│              │   Unified API      │                         │
│              │   (Flask Server)   │                         │
│              └─────────┬──────────┘                         │
│                        │                                     │
│         ┌──────────────┼──────────────┐                     │
│         │              │              │                     │
│    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐               │
│    │   n8n   │   │ Docker  │   │ Direct  │               │
│    │ Flows   │   │ Deploy  │   │  REST   │               │
│    └─────────┘   └─────────┘   └─────────┘               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.8+ (recommended: Python 3.11)
- OpenAI API key
- Node.js (optional, for n8n)
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd unified-product-system

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy env_example.txt .env
# Edit .env and add your OPENAI_API_KEY
```

### Start the Unified API

**Windows:**

```bash
start_unified_api.bat
```

**Linux/Mac:**

```bash
python web_service.py
```

The API will be available at `http://localhost:5000`

---

## Installation

### Manual Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Install specific module dependencies (optional)
pip install -r requirements_api.txt      # API only
pip install -r requirements_tags.txt     # Tagging only
pip install -r requirements_translation.txt  # Translation only
```

### Environment Setup

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
FLASK_ENV=production
```

### Verify Installation

```bash
# Test translation API
python web_service.py

# Test tagging API
python api_server.py
```

---

## Usage

### Product Tagging

#### One-Time Setup (Generate Taxonomy Index)

```bash
# Step 1: Generate descriptors from taxonomy
python gen_descriptors.py \
  --inp "texo/FULL TEXO.xlsx" \
  --out "texo/taxonomy_descriptors_texo.xlsx" \
  --model "gpt-4o-mini" \
  --batch 12

# Step 2: Create embeddings index
python KNN_retriver.py \
  --inp "texo/taxonomy_descriptors_texo.xlsx" \
  --out "texo/node_embeddings_texo.parquet" \
  --model "text-embedding-3-large" \
  --batch 64 \
  --faiss
```

#### Tag Products

```bash
# Step 3: Generate pseudo labels for products
python bootstrap_pseudolabels.py \
  --nodes "texo/node_embeddings_texo.parquet" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --products "YOUR_PRODUCTS.xlsx" \
  --title-col "Title" \
  --desc-col "Body (HTML)" \
  --out "pseudo_labels.csv"

# Step 4: Format for upload
python unified_tagging_pipeline.py \
  --pseudo "pseudo_labels.csv" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --data "YOUR_PRODUCTS.xlsx" \
  --output-dir "output"
```

**Output Files:**

- `output/pseudo_labels_with_tags.csv` - Products with assigned tags
- `output/merged_data_with_tags.xlsx` - Complete merged data
- `output/upload_template_final.xlsx` - Ready for e-commerce upload

### Translation & SEO

```bash
# Basic usage
python universal_translation_seo.py \
  --input "products.xlsx" \
  --output "translated_products.xlsx" \
  --name-col "Title" \
  --desc-col "Body (HTML)" \
  --sample 100

# With custom product context
python universal_translation_seo.py \
  --input "tools.xlsx" \
  --output "tools_arabic.xlsx" \
  --product-context "power tools and hardware"
```

**Generated Columns:**

- `arabic description` - Full Arabic product description
- `product_title_ar` - Arabic product title (50-80 chars)
- `meta_title` - SEO meta title (≤60 chars)
- `meta_description` - SEO meta description (≤155 chars)
- `handle_ar` - SEO-friendly URL handle

### Unified API

Start the API server:

```bash
python web_service.py
```

#### API Endpoints

**Health Check:**

```bash
curl http://localhost:5000/health
```

**Translation:**

```bash
curl -X POST http://localhost:5000/process \
  -H "Content-Type: application/json" \
  -d '{
    "csv_content": "Title,Body (HTML)\nProduct 1,Description 1",
    "product_context": "tools",
    "sample": 5
  }'
```

**Download Translated File:**

```bash
curl http://localhost:5000/download/<file_id> -o output.xlsx
```

---

## 📡 API Reference

### Translation API (`web_service.py`)

#### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "translation-web-service"
}
```

#### `POST /process`

Process CSV data for translation and SEO.

**Request Body:**

```json
{
  "csv_content": "Title,Body (HTML)\nProduct Name,Description",
  "product_context": "tools and equipment",
  "sample": 5
}
```

**Response:**

```json
{
  "success": true,
  "message": "Translation and SEO processing completed successfully",
  "file_id": "uuid-here",
  "output_columns": ["arabic description", "product_title_ar", "meta_title", "meta_description"]
}
```

#### `GET /download/<file_id>`

Download processed file.

**Response:** Excel file download

### Tagging API (`api_server.py`)

#### `GET /api/health`

Health check with taxonomy status.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-10-01T12:00:00",
  "taxonomy_ready": true
}
```

#### `POST /api/setup-taxonomy`

One-time taxonomy setup (generates descriptors and embeddings).

**Request Body:**

```json
{
  "taxonomy_file": "texo/FULL TEXO.xlsx"
}
```

#### `POST /api/tag-products-simple`

Tag products using JSON data.

**Request Body:**

```json
{
  "products_data": [
    {
      "Title": "Product Name",
      "Body (HTML)": "Product description",
      "Product Type": "Category"
    }
  ]
}
```

**Response:**

```json
{
  "status": "success",
  "data": [
    {
      "tags": "tools,hand-tools,tool-kits",
      "Title": "Product Name",
      ...
    }
  ],
  "stats": {
    "total_products": 1,
    "tagged_products": 1
  }
}
```

---

## n8n Integration

### Available Workflows

The system includes three pre-built n8n workflows:

1. **`complete-workflow.json`** - Full pipeline (tagging + translation)
2. **`translation-only.json`** - Translation and SEO only
3. **`unified-workflow.json`** - Unified workflow with Google Sheets

### Setup n8n with Docker

```bash
# Start both API and n8n
docker-compose up -d

# Access n8n at http://localhost:5678
# Default credentials: admin / password
```

### Import Workflows

1. Open n8n at `http://localhost:5678`
2. Click **Workflows** → **Import from File**
3. Select workflow JSON from `n8n_workflows/`
4. Configure Google Sheets credentials
5. Update API endpoints if needed
6. Activate and run!

### Google Sheets Integration

**Required Columns:**

- `Title` (English product name)
- `Body (HTML)` (English description)

**Optional Columns:**

- `Brand`
- `Category`
- `SKU`
- `Product Type`

**Workflow Steps:**

1. Manual trigger or webhook
2. Read data from Google Sheets
3. Convert to CSV format
4. Send to translation API
5. Tag products (optional)
6. Download results
7. Write back to Google Sheets (optional)

---

## Docker Deployment

### Quick Start with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Services

- **Tagging API**: `http://localhost:5001`
- **n8n**: `http://localhost:5678`

### Custom Docker Build

```bash
# Build image
docker build -t unified-product-system .

# Run container
docker run -d \
  -p 5000:5000 \
  -e OPENAI_API_KEY=your_key_here \
  -v $(pwd)/texo:/app/texo \
  -v $(pwd)/output:/app/output \
  unified-product-system
```

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key
FLASK_ENV=production
PORT=5000
```

---

## Configuration

### OpenAI Models

**Tagging System:**

- Embedding Model: `text-embedding-3-large` (3072 dimensions)
- LLM Model: `gpt-4o-mini`
- Confidence Threshold: 0.85
- Top-K Candidates: 8

**Translation System:**

- LLM Model: `gpt-4o-mini`
- Async Workers: 5
- Max Retries: 3

### Performance Tuning

**Batch Processing:**

```python
# gen_descriptors.py
--batch 12  # Number of parallel API calls

# KNN_retriver.py
--batch 64  # Embedding batch size
```

**API Settings:**

```python
# web_service.py
port = 5000
workers = 5  # Async workers
```

---

## File Structure

```
unified-product-system/
├── apis/                          # API modules
│   ├── bootstrap_pseudolabels.py  # Product tagging logic
│   ├── gen_descriptors.py         # Taxonomy descriptor generation
│   ├── KNN_retriver.py            # FAISS similarity search
│   ├── tagging_api.py             # Tagging API endpoints
│   ├── translation_api.py         # Translation API endpoints
│   ├── unified_tagging_pipeline.py # Complete tagging pipeline
│   └── universal_translation_seo.py # Translation engine
│
├── data/texo/                     # Taxonomy data
│   ├── FULL TEXO.xlsx             # Original taxonomy
│   ├── node_embeddings_texo.parquet # Vector embeddings
│   └── taxonomy_descriptors_texo.xlsx # Generated descriptors
│
├── n8n_workflows/                 # n8n automation workflows
│   ├── complete-workflow.json     # Full pipeline
│   ├── translation-only.json      # Translation only
│   └── unified-workflow.json      # Unified workflow
│
├── output/                        # Generated output files
│   ├── merged_data_with_tags.xlsx
│   ├── pseudo_labels_with_tags.csv
│   └── upload_template_final.xlsx
│
├── api_server.py                  # Tagging API server
├── web_service.py                 # Translation API server
├── start_apis.py                  # Start both APIs
├── start_unified_api.bat          # Windows startup script
│
├── docker-compose.yml             # Docker orchestration
├── Dockerfile                     # Container definition
│
├── requirements.txt               # All dependencies
├── requirements_api.txt           # API dependencies
├── requirements_tags.txt          # Tagging dependencies
├── requirements_translation.txt   # Translation dependencies
│
├── README.md                      # This file
├── README_tags.md                 # Tagging system docs
├── README_translation.md          # Translation system docs
│
└── .env                           # Environment variables (create this)
```

---

## Troubleshooting

### Common Issues

#### 1. API Key Not Set

```bash
Error: OpenAI API key not found
```

**Solution:** Create `.env` file with your API key:

```bash
OPENAI_API_KEY=sk-your-key-here
```

#### 2. Port Already in Use

```bash
Error: Address already in use
```

**Solution:** Change port in code or kill existing process:

```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <pid> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

#### 3. Missing Dependencies

```bash
Error: No module named 'faiss'
```

**Solution:** Install missing packages:

```bash
pip install -r requirements.txt
```

#### 4. Taxonomy Not Ready

```bash
Error: Taxonomy not ready
```

**Solution:** Run setup first:

```bash
curl -X POST http://localhost:5001/api/setup-taxonomy
```

#### 5. Google Sheets Permission Denied

```bash
Error: Insufficient authentication scopes
```

**Solution:** Reconfigure OAuth2 in n8n with correct scopes:

- `https://www.googleapis.com/auth/spreadsheets`
- `https://www.googleapis.com/auth/drive.file`

#### 6. Memory Issues with Large Files

**Solution:** Process in smaller batches:

```bash
python bootstrap_pseudolabels.py --batch 32 ...
```

### Debug Mode

Enable detailed logging:

```bash
# Python scripts
python script.py --log-level DEBUG

# Flask API
export FLASK_ENV=development
python web_service.py
```

### Check Service Health

```bash
# Translation API
curl http://localhost:5000/health

# Tagging API
curl http://localhost:5001/api/health

# Check taxonomy status
curl http://localhost:5001/api/status
```

---

## Performance Metrics

### Typical Processing Speeds

| Operation             | Speed                | Notes                      |
| --------------------- | -------------------- | -------------------------- |
| Translation           | 100-200 products/min | Depends on API rate limits |
| Tagging               | 200-500 products/min | With FAISS acceleration    |
| Embedding Generation  | 1000-2000 texts/min  | Batch size: 64             |
| Descriptor Generation | 50-100 nodes/min     | Batch size: 12             |

### Optimization Tips

1. **Use batch processing** for large datasets
2. **Enable FAISS indexing** for faster similarity search
3. **Increase async workers** for better throughput
4. **Cache embeddings** to avoid regeneration
5. **Use Docker** for consistent performance

---

## Testing

### Test Translation API

```bash
# Start server
python web_service.py

# Test endpoint
curl http://localhost:5000/test
```

### Test Tagging API

```bash
# Start server
python api_server.py

# Check health
curl http://localhost:5001/api/health

# Run test script
python test_unified_api.py
```

### Test Docker Setup

```bash
# Test Docker integration
python test_docker_integration.py

# Test complete workflow
docker-compose up -d
curl http://localhost:5001/api/health
```

---

## Additional Documentation

- **[Tagging System Details](README_tags.md)** - Complete tagging workflow
- **[Translation System Details](README_translation.md)** - Translation & SEO guide
- **[n8n Workflows](n8n_workflows/)** - Automation examples
- **Environment Setup**: See `env_example.txt`

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to functions
- Include type hints where possible
- Write unit tests for new features
- Update documentation

---

## Roadmap

- [ ] Multi-language support (beyond Arabic)
- [ ] Real-time processing via WebSockets
- [ ] Advanced analytics dashboard
- [ ] Shopify/WooCommerce plugins
- [ ] Batch API for enterprise clients
- [ ] Machine learning model fine-tuning
- [ ] Automated quality scoring
#   N 8 N P i p e L i n e - t a g s A n d 
 
 