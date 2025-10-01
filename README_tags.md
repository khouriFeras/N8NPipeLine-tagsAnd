#  Automated Product Tagging System

A complete AI-powered pipeline for automatically generating hierarchical taxonomy tags for products using OpenAI's GPT models and embedding-based similarity search.

##  Overview

This system takes raw product data and automatically assigns appropriate taxonomy tags by:
1. **Generating bilingual descriptors** for taxonomy nodes (EN+AR)
2. **Creating embeddings** for semantic similarity search
3. **Using AI to match** products to the most relevant taxonomy nodes
4. **Formatting output** for easy upload to e-commerce platforms

##  Complete Workflow

### **Phase 1: Setup & Preparation**

#### Step 1: Generate Descriptors
```bash
python gen_descriptors.py \
  --inp "texo/FULL TEXO.xlsx" \
  --sheet "Tags" \
  --l1 "L1 Tags " \
  --l2 "L2 Tags " \
  --l3 "L3 Tags " \
  --l4 "L4 Tags " \
  --out "texo/taxonomy_descriptors_texo.xlsx" \
  --model "gpt-4o-mini" \
  --batch 12
```

**What it does:** Generates bilingual (English + Arabic) descriptions for each taxonomy node using OpenAI.

#### Step 2: Create Embeddings Index
```bash
python KNN_retriver.py \
  --inp "texo/taxonomy_descriptors_texo.xlsx" \
  --sheet "descriptors" \
  --out "texo/node_embeddings_texo.parquet" \
  --model "text-embedding-3-large" \
  --batch 64 \
  --faiss
```

**What it does:** Creates vector embeddings for all taxonomy nodes to enable fast similarity search.

### **Phase 2: Product Tagging**

#### Step 3: Generate Pseudo Labels
```bash
python bootstrap_pseudolabels.py \
  --nodes "texo/node_embeddings_texo.parquet" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --sheet "descriptors" \
  --products "YOUR_PRODUCTS.xlsx" \
  --prod-sheet "Sheet1" \
  --title-col "Title" \
  --desc-col "Body (HTML)" \
  --embed-model "text-embedding-3-large" \
  --llm-model "gpt-4o-mini" \
  --topk 8 \
  --conf-thresh 0.85 \
  --out "pseudo_labels.csv"
```

**What it does:** Uses AI to match each product to the most relevant taxonomy node based on product title and description.

#### Step 4: Final Formatting (Unified Pipeline)
```bash
python unified_tagging_pipeline.py \
  --pseudo "pseudo_labels.csv" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --data "YOUR_PRODUCTS.xlsx" \
  --output-dir "output"
```

**What it does:** Adds hierarchical tags and formats the output for easy upload to e-commerce platforms.

##  Workflow Diagram

```
Raw Taxonomy (FULL TEXO.xlsx)
    ↓ [gen_descriptors.py]
Bilingual Descriptors
    ↓ [KNN_retriver.py]
Embeddings Index
    ↓ [bootstrap_pseudolabels.py]
Your Products → Pseudo Labels
    ↓ [unified_tagging_pipeline.py]
Final Upload Template
```

##  Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Required packages (see `requirements.txt`)

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd tagsFinal

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### One-Time Setup (Run Once)
```bash
# Step 1: Generate descriptors
python gen_descriptors.py --inp "texo/FULL TEXO.xlsx" --out "texo/taxonomy_descriptors_texo.xlsx"

# Step 2: Create embeddings
python KNN_retriver.py --inp "texo/taxonomy_descriptors_texo.xlsx" --out "texo/node_embeddings_texo.parquet" --faiss
```

### Process New Products
```bash
# Step 3: Generate pseudo labels
python bootstrap_pseudolabels.py \
  --nodes "texo/node_embeddings_texo.parquet" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --products "YOUR_PRODUCTS.xlsx" \
  --out "pseudo_labels.csv"

# Step 4: Format for upload
python unified_tagging_pipeline.py \
  --pseudo "pseudo_labels.csv" \
  --descriptors "texo/taxonomy_descriptors_texo.xlsx" \
  --data "YOUR_PRODUCTS.xlsx" \
  --output-dir "output"
```

##  File Structure

```
tagsFinal/
├── texo/
│   ├── FULL TEXO.xlsx                    # Original taxonomy
│   ├── taxonomy_descriptors_texo.xlsx    # Generated descriptors
│   └── node_embeddings_texo.parquet     # Embeddings index
├── gen_descriptors.py                    # Step 1: Generate descriptors
├── KNN_retriver.py                       # Step 2: Create embeddings
├── bootstrap_pseudolabels.py             # Step 3: Generate pseudo labels
├── unified_tagging_pipeline.py           # Step 4: Format output
├── requirements.txt                      # Dependencies
└── README.md                            # This file
```

##  Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `gen_descriptors.py` | Generate bilingual descriptors | Raw taxonomy | Descriptors Excel |
| `KNN_retriver.py` | Create embeddings index | Descriptors | Parquet + FAISS |
| `bootstrap_pseudolabels.py` | Match products to taxonomy | Products + Index | Pseudo labels CSV |
| `unified_tagging_pipeline.py` | Format final output | Pseudo labels + Products | Upload template |

## 📊 Example Output

### Input Product
```
Title: "طقم بوكس علبة حديد(40) RONIX"
Product Type: "Tools"
```

### Generated Tags
```
tools,hand-tools,tool-kits
```

### Final Upload Template
| tags | Title | Product Type | ... |
|------|-------|--------------|-----|
| tools,hand-tools,tool-kits | طقم بوكس علبة حديد(40) RONIX | Tools | ... |

## ⚙️ Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Model Settings
- **Embedding Model**: `text-embedding-3-large` (3072 dimensions)
- **LLM Model**: `gpt-4o-mini` (for classification)
- **Batch Size**: 12-64 (adjustable based on API limits)

### Performance Tuning
- **Confidence Threshold**: 0.85 (minimum confidence to accept)
- **Top-K Candidates**: 8 (number of candidates to consider)
- **Auto-accept Threshold**: 0.82 (skip LLM for high-confidence matches)

## 🔧 Advanced Usage

### Individual Steps
```bash
# Run only descriptor generation
python gen_descriptors.py --inp "taxonomy.xlsx" --out "descriptors.xlsx"

# Run only pseudo labeling
python bootstrap_pseudolabels.py --nodes "embeddings.parquet" --products "products.xlsx" --out "labels.csv"

# Run only formatting
python unified_tagging_pipeline.py --step format-upload --pseudo "labels.csv" --data "products.xlsx" --output "final.xlsx"
```

### Custom Column Mapping
```bash
python bootstrap_pseudolabels.py \
  --title-col "Product Name" \
  --desc-col "Description" \
  --products "custom_products.xlsx"
```

##  Performance

- **Processing Speed**: ~100-500 products per minute (depending on API limits)
- **Accuracy**: 85-95% verified matches (with semantic verification)
- **Languages**: Supports English and Arabic product descriptions
- **Scalability**: Handles thousands of products efficiently

##  Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Column Not Found**
   - Check column names in your input files
   - Use `--title-col` and `--desc-col` to specify correct columns

3. **Memory Issues**
   - Reduce batch size: `--batch 32`
   - Process files in smaller chunks

4. **No Matches Found**
   - Check if product titles are in the correct language
   - Verify taxonomy descriptors are generated correctly

## 🔌 n8n Integration

This system is n8n-ready with a complete REST API for easy workflow automation.

### Quick Start with n8n

1. **Start the API Server**:
   ```bash
   # Linux/Mac
   ./start_api.sh
   
   # Windows
   start_api.bat
   
   # Or with Docker
   docker-compose up
   ```

2. **Setup Taxonomy** (one-time):
   ```bash
   curl -X POST http://localhost:5000/api/setup-taxonomy
   ```

3. **Tag Products via API**:
   ```bash
   curl -X POST http://localhost:5000/api/tag-products-simple \
     -H "Content-Type: application/json" \
     -d '{"products_data": [{"Title": "Product Name", "Product Type": "Category"}]}'
   ```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/status` | System status |
| POST | `/api/setup-taxonomy` | Setup taxonomy (one-time) |
| POST | `/api/tag-products` | Tag products from file |
| POST | `/api/tag-products-simple` | Tag products from JSON |
| GET | `/api/download/<filename>` | Download files |

### n8n Workflow Examples

**Simple Product Tagging**:
```json
{
  "nodes": [
    {
      "name": "HTTP Request",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "method": "POST",
        "url": "http://localhost:5000/api/tag-products-simple",
        "body": {
          "products_data": "={{ $json }}"
        }
      }
    }
  ]
}
```

**E-commerce Integration**:
1. Trigger: New product added
2. Transform: Convert to tagging format
3. Tag: Send to API
4. Update: Apply tags to product

See `n8n_workflows/README.md` for complete integration guide.

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access n8n at http://localhost:5678
# Access API at http://localhost:5000
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the script help: `python script_name.py --help`
- Check n8n integration guide: `n8n_workflows/README.md`

---

**Built with ❤️ for automated product tagging and n8n integration**
