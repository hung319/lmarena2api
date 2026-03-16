#!/usr/bin/env python3
"""
Test script to verify the fix for the sync iterator on async stream error
"""

from openai import OpenAI

# Configuration - using the same as in the main chat script
BASE_URL = "http://localhost:8000/v1"
API_KEY = "sk-lmab-4d4c13f6-7846-4f94-a261-f59911838196"


def test_sync_streaming():
    print("Testing sync streaming with OpenAI v2.x...")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # First, get available models
    try:
        models = client.models.list()
        print(f"✓ Successfully fetched {len(models.data)} models")

        if models.data:
            # Use the first available model for testing
            test_model = models.data[0].id
            print(f"Using model: {test_model}")

            # Test streaming
            print("Testing stream iteration...")
            stream = client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": "hello"}],
                stream=True,
            )

            # This is the critical part - make sure we can iterate sync over the stream
            content = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)

            print("\n✓ Sync streaming iteration successful!")
            print(f"Received content length: {len(content)}")
            return True
        else:
            print("No models available for testing")
            return False

    except Exception as e:
        print(f"✗ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_sync_streaming()
    if success:
        print("\n✓ Test PASSED: Sync streaming works correctly with OpenAI v2.x")
    else:
        print("\n✗ Test FAILED: There are still issues with streaming")
