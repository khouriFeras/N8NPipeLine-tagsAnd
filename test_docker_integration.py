#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Docker Integration Script

This script tests the product tagging API after Docker integration.
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_status():
    """Test status endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed")
            print(f"   Taxonomy ready: {data['taxonomy_ready']}")
            print(f"   Available endpoints: {len(data['endpoints'])}")
            return True
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def test_setup_taxonomy():
    """Test taxonomy setup."""
    try:
        print("🔄 Setting up taxonomy...")
        response = requests.post(
            f"{API_BASE_URL}/api/setup-taxonomy",
            json={"taxonomy_file": "texo/FULL TEXO.xlsx"},
            timeout=300  # 5 minutes timeout for setup
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Taxonomy setup completed: {data['message']}")
            return True
        else:
            print(f"❌ Taxonomy setup failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Taxonomy setup error: {e}")
        return False

def test_tag_products():
    """Test product tagging."""
    try:
        test_products = [
            {
                "Title": "طقم بوكس علبة حديد(40) RONIX",
                "Product Type": "Tools",
                "Vendor": "RONIX",
                "Body (HTML)": "Iron tool box set for professional use"
            },
            {
                "Title": "مضخة غسيل 100 بار RONIX",
                "Product Type": "Water Pumps",
                "Vendor": "RONIX",
                "Body (HTML)": "High pressure water pump for cleaning"
            }
        ]
        
        print("🔄 Testing product tagging...")
        response = requests.post(
            f"{API_BASE_URL}/api/tag-products-simple",
            json={"products_data": test_products},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Product tagging successful")
            print(f"   Total products: {data['stats']['total_products']}")
            print(f"   Tagged products: {data['stats']['tagged_products']}")
            
            # Show sample results
            if data['data']:
                print("\n📋 Sample results:")
                for i, product in enumerate(data['data'][:2], 1):
                    print(f"   {i}. {product['Title']}")
                    print(f"      Tags: {product.get('tags', 'No tags')}")
                    print(f"      Confidence: {product.get('confidence', 'N/A')}")
            
            return True
        else:
            print(f"❌ Product tagging failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Product tagging error: {e}")
        return False

def main():
    print("🧪 Docker Integration Test")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    if not test_health():
        print("❌ Health check failed. Is the API server running?")
        print("   Run: docker-compose up -d tagging-api")
        sys.exit(1)
    
    # Test 2: Status check
    print("\n2️⃣ Testing status endpoint...")
    if not test_status():
        print("❌ Status check failed")
        sys.exit(1)
    
    # Test 3: Setup taxonomy (if not ready)
    print("\n3️⃣ Testing taxonomy setup...")
    if not test_setup_taxonomy():
        print("❌ Taxonomy setup failed")
        sys.exit(1)
    
    # Wait a moment for setup to complete
    print("⏳ Waiting for setup to complete...")
    time.sleep(5)
    
    # Test 4: Product tagging
    print("\n4️⃣ Testing product tagging...")
    if not test_tag_products():
        print("❌ Product tagging failed")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Docker integration is working correctly.")
    print("\n📋 Your tagging API is ready to use:")
    print(f"   Health: {API_BASE_URL}/api/health")
    print(f"   Status: {API_BASE_URL}/api/status")
    print(f"   Tag products: {API_BASE_URL}/api/tag-products-simple")
    
    print("\n🔌 n8n Integration:")
    print("   Use the HTTP Request node in n8n to call the API endpoints")
    print("   See n8n_workflows/README.md for detailed examples")

if __name__ == "__main__":
    main()
