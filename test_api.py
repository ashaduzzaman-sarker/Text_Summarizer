# ============================================================================
# test_api.py (API Testing Script)
# ============================================================================
"""Test script for FastAPI endpoints."""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Test article
TEST_ARTICLE = """
(CNN) -- Usain Bolt rounded off the world championships Sunday by claiming his third gold in Moscow as he anchored Jamaica to victory in the men's 4x100m relay. The fastest man in the world charged clear of United States rival Justin Gatlin as the Jamaican quartet of Nesta Carter, Kemar Bailey-Cole, Nickel Ashmeade and Bolt won in 37.36 seconds. The U.S finished second in 37.56 seconds with Canada taking the bronze after Britain were disqualified for a faulty handover.
"""


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_summarize():
    """Test summarization endpoint."""
    print("Testing /summarize endpoint...")
    
    payload = {
        "text": TEST_ARTICLE,
        "max_length": 128,
        "min_length": 30,
        "num_beams": 4
    }
    
    response = requests.post(
        f"{BASE_URL}/summarize",
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {result['summary']}")
        print(f"Input Length: {result['input_length']} words")
        print(f"Summary Length: {result['summary_length']} words\n")
    else:
        print(f"Error: {response.text}\n")


if __name__ == "__main__":
    test_health()
    test_summarize()