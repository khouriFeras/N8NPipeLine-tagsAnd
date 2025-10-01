#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the unified API
Tests both translation and tagging functionality
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_service_info():
    """Test service info endpoint"""
    print("\n🔍 Testing service info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/test")
        print(f"✅ Service info: {response.status_code}")
        print(f"   Response: {response.json()}")
        return True
    except Exception as e:
        print(f"❌ Service info failed: {e}")
        return False

def test_translation():
    """Test translation endpoint"""
    print("\n🔍 Testing translation endpoint...")
    
    test_data = {
        "csv_content": "Title,Body (HTML)\nTest Product,This is a test product description",
        "product_context": "test equipment",
        "sample": 1
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/process",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Translation test: {response.status_code}")
        result = response.json()
        print(f"   Success: {result.get('success', False)}")
        if result.get('success'):
            print(f"   File ID: {result.get('file_id')}")
            print(f"   Download URL: {result.get('download_url')}")
        else:
            print(f"   Error: {result.get('error')}")
        return result.get('success', False)
    except Exception as e:
        print(f"❌ Translation test failed: {e}")
        return False

def test_tagging():
    """Test tagging endpoint"""
    print("\n🔍 Testing tagging endpoint...")
    
    test_data = {
        "products_data": [
            {
                "Title": "Test Product",
                "Product Type": "Electronics",
                "Vendor": "Test Brand",
                "Body (HTML)": "This is a test product description for tagging"
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/tag-products-simple",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Tagging test: {response.status_code}")
        result = response.json()
        print(f"   Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"   Tagged products: {result.get('stats', {}).get('tagged_products', 0)}")
        else:
            print(f"   Message: {result.get('message')}")
        return result.get('status') == 'success'
    except Exception as e:
        print(f"❌ Tagging test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Unified API Tests")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health()
    info_ok = test_service_info()
    
    if not health_ok or not info_ok:
        print("\n❌ Basic tests failed. Make sure the API is running:")
        print("   cd D:\\JafarShop\\unified-product-system")
        print("   python unified_api.py")
        return
    
    # Test translation
    translation_ok = test_translation()
    
    # Test tagging (might fail if taxonomy not set up)
    tagging_ok = test_tagging()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Health Check: {'✅' if health_ok else '❌'}")
    print(f"   Service Info: {'✅' if info_ok else '❌'}")
    print(f"   Translation:  {'✅' if translation_ok else '❌'}")
    print(f"   Tagging:      {'✅' if tagging_ok else '❌'}")
    
    if health_ok and info_ok and translation_ok:
        print("\n🎉 Core functionality working! Ready for deployment.")
        if not tagging_ok:
            print("⚠️  Tagging failed - this is normal if taxonomy isn't set up yet.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

