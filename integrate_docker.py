#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Docker Integration Helper Script

This script helps integrate the product tagging system with your existing Docker setup.
"""

import os
import sys
import json
import yaml
from pathlib import Path

def find_docker_compose_files():
    """Find existing docker-compose files."""
    possible_files = [
        'docker-compose.yml',
        'docker-compose.yaml',
        'compose.yml',
        'compose.yaml'
    ]
    
    found_files = []
    for file in possible_files:
        if Path(file).exists():
            found_files.append(file)
    
    return found_files

def read_docker_compose(file_path):
    """Read and parse docker-compose file."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def add_tagging_service(compose_data):
    """Add tagging service to docker-compose data."""
    if 'services' not in compose_data:
        compose_data['services'] = {}
    
    tagging_service = {
        'tagging-api': {
            'build': '.',
            'container_name': 'product-tagging-api',
            'ports': ['5000:5000'],
            'environment': [
                'OPENAI_API_KEY=${OPENAI_API_KEY}',
                'FLASK_ENV=production'
            ],
            'volumes': [
                './texo:/app/texo',
                './output:/app/output',
                './descriptor_cache.jsonl:/app/descriptor_cache.jsonl',
                './emb_cache.jsonl:/app/emb_cache.jsonl'
            ],
            'restart': 'unless-stopped',
            'healthcheck': {
                'test': ['CMD', 'curl', '-f', 'http://localhost:5000/api/health'],
                'interval': '30s',
                'timeout': '10s',
                'retries': 3,
                'start_period': '40s'
            }
        }
    }
    
    compose_data['services'].update(tagging_service)
    return compose_data

def write_docker_compose(file_path, compose_data):
    """Write docker-compose data to file."""
    try:
        with open(file_path, 'w') as f:
            yaml.dump(compose_data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        print(f"Error writing {file_path}: {e}")
        return False

def create_env_file():
    """Create .env file with required environment variables."""
    env_content = """# Product Tagging API Environment Variables
OPENAI_API_KEY=your-openai-api-key-here
FLASK_ENV=production
FLASK_PORT=5000

# Optional: Add your existing environment variables below
"""
    
    if not Path('.env').exists():
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please update OPENAI_API_KEY")
    else:
        print("⚠️  .env file already exists - please add OPENAI_API_KEY if not present")

def main():
    print("🐳 Docker Integration Helper")
    print("=" * 40)
    
    # Find existing docker-compose files
    compose_files = find_docker_compose_files()
    
    if not compose_files:
        print("❌ No docker-compose files found in current directory")
        print("Please run this script in the directory containing your docker-compose.yml")
        return
    
    print(f"📁 Found docker-compose files: {', '.join(compose_files)}")
    
    # Use the first found file
    main_compose_file = compose_files[0]
    print(f"📝 Using: {main_compose_file}")
    
    # Read existing docker-compose
    compose_data = read_docker_compose(main_compose_file)
    if not compose_data:
        return
    
    # Check if tagging service already exists
    if 'tagging-api' in compose_data.get('services', {}):
        print("⚠️  Tagging service already exists in docker-compose.yml")
        response = input("Do you want to update it? (y/N): ")
        if response.lower() != 'y':
            print("Skipping docker-compose update")
        else:
            # Update existing service
            compose_data = add_tagging_service(compose_data)
            if write_docker_compose(main_compose_file, compose_data):
                print(f"✅ Updated {main_compose_file}")
            else:
                print(f"❌ Failed to update {main_compose_file}")
    else:
        # Add new service
        compose_data = add_tagging_service(compose_data)
        if write_docker_compose(main_compose_file, compose_data):
            print(f"✅ Added tagging service to {main_compose_file}")
        else:
            print(f"❌ Failed to update {main_compose_file}")
    
    # Create .env file
    create_env_file()
    
    # Create necessary directories
    directories = ['texo', 'output']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    print("\n🚀 Next Steps:")
    print("1. Update .env file with your OPENAI_API_KEY")
    print("2. Run: docker-compose up -d tagging-api")
    print("3. Setup taxonomy: curl -X POST http://localhost:5000/api/setup-taxonomy")
    print("4. Test API: curl http://localhost:5000/api/health")
    
    print("\n📋 Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/setup-taxonomy - Setup taxonomy (one-time)")
    print("  POST /api/tag-products-simple - Tag products")
    print("  GET  /api/status - System status")

if __name__ == "__main__":
    main()
