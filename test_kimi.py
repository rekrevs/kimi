#!/usr/bin/env python
"""Comprehensive testing suite for Kimi-K2.5 deployment."""

import json
import time
import requests
import base64
from pathlib import Path

API_URL = "http://localhost:31245/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

def chat(messages, max_tokens=500, temperature=0.7):
    """Send a chat request and return response with timing."""
    payload = {
        "model": "Kimi-K2.5",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    start = time.time()
    resp = requests.post(API_URL, headers=HEADERS, json=payload)
    elapsed = time.time() - start

    if resp.status_code != 200:
        return None, 0, 0, elapsed, resp.text

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data["usage"]
    prompt_tokens = usage["prompt_tokens"]
    completion_tokens = usage["completion_tokens"]

    return content, prompt_tokens, completion_tokens, elapsed, None

def measure_tokens_per_second(runs=5):
    """Measure generation speed across multiple runs."""
    print("\n" + "="*60)
    print("TEST 1: Tokens per Second Measurement")
    print("="*60)

    prompt = "Write a detailed explanation of how neural networks learn through backpropagation. Include the mathematical concepts involved."

    results = []
    for i in range(runs):
        content, prompt_tok, comp_tok, elapsed, err = chat(
            [{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )

        if err:
            print(f"  Run {i+1}: ERROR - {err[:100]}")
            continue

        tps = comp_tok / elapsed
        results.append({
            "run": i+1,
            "prompt_tokens": prompt_tok,
            "completion_tokens": comp_tok,
            "elapsed_seconds": round(elapsed, 2),
            "tokens_per_second": round(tps, 2)
        })
        print(f"  Run {i+1}: {comp_tok} tokens in {elapsed:.2f}s = {tps:.2f} tok/s")

    if results:
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        print(f"\n  AVERAGE: {avg_tps:.2f} tokens/second")
        return {"runs": results, "average_tps": round(avg_tps, 2)}
    return None

def test_large_context():
    """Test with large context input."""
    print("\n" + "="*60)
    print("TEST 2: Large Context Handling")
    print("="*60)

    # Generate a long document (~10K tokens worth of text)
    long_text = """
    The history of artificial intelligence is a fascinating journey through decades of research,
    breakthroughs, and setbacks. From the early days at Dartmouth in 1956 to the modern era of
    large language models, the field has evolved dramatically.
    """ * 200  # Repeat to create ~10K tokens

    long_text += "\n\nBased on the text above, provide a brief 2-sentence summary."

    content, prompt_tok, comp_tok, elapsed, err = chat(
        [{"role": "user", "content": long_text}],
        max_tokens=100,
        temperature=0.3
    )

    result = {
        "prompt_tokens": prompt_tok,
        "completion_tokens": comp_tok,
        "elapsed_seconds": round(elapsed, 2),
        "success": err is None
    }

    if err:
        print(f"  ERROR: {err[:200]}")
        result["error"] = err[:200]
    else:
        print(f"  Input: {prompt_tok} tokens")
        print(f"  Output: {comp_tok} tokens in {elapsed:.2f}s")
        print(f"  Response: {content[:200]}...")
        result["response_preview"] = content[:200]

    return result

def test_coding_tasks():
    """Test coding capabilities."""
    print("\n" + "="*60)
    print("TEST 3: Coding Capabilities")
    print("="*60)

    tests = []

    # Test 3a: Code generation
    print("\n  3a. Code Generation - Binary Search")
    content, _, _, elapsed, err = chat([{
        "role": "user",
        "content": "Write a Python function for binary search that returns the index of the target element, or -1 if not found. Include type hints."
    }], max_tokens=300, temperature=0.3)

    if not err:
        print(f"    Generated in {elapsed:.2f}s")
        print(f"    Response preview: {content[:150]}...")
        # Check for key elements
        has_def = "def " in content
        has_return = "return" in content
        has_type_hints = "->" in content or ": int" in content or ": list" in content.lower()
        tests.append({
            "test": "binary_search_generation",
            "success": has_def and has_return,
            "has_type_hints": has_type_hints,
            "elapsed": round(elapsed, 2)
        })
    else:
        tests.append({"test": "binary_search_generation", "success": False, "error": err[:100]})

    # Test 3b: Debug code
    print("\n  3b. Code Debugging")
    buggy_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-3)  # Bug here

print(fibonacci(10))  # Should print 55
'''
    content, _, _, elapsed, err = chat([{
        "role": "user",
        "content": f"Find and fix the bug in this code:\n```python\n{buggy_code}\n```"
    }], max_tokens=300, temperature=0.3)

    if not err:
        print(f"    Analyzed in {elapsed:.2f}s")
        found_bug = "n-2" in content or "n - 2" in content or "fibonacci(n-2)" in content
        print(f"    Found the bug (n-3 should be n-2): {found_bug}")
        tests.append({
            "test": "debug_fibonacci",
            "success": found_bug,
            "elapsed": round(elapsed, 2)
        })
    else:
        tests.append({"test": "debug_fibonacci", "success": False, "error": err[:100]})

    # Test 3c: Explain algorithm
    print("\n  3c. Algorithm Explanation")
    content, _, _, elapsed, err = chat([{
        "role": "user",
        "content": "Explain the quicksort algorithm. What is its average and worst-case time complexity?"
    }], max_tokens=400, temperature=0.3)

    if not err:
        print(f"    Explained in {elapsed:.2f}s")
        has_nlogn = "n log n" in content.lower() or "nlogn" in content.lower() or "O(n log n)" in content
        has_n2 = "n^2" in content or "n²" in content or "O(n^2)" in content or "O(n2)" in content
        print(f"    Mentions O(n log n) average: {has_nlogn}")
        print(f"    Mentions O(n²) worst case: {has_n2}")
        tests.append({
            "test": "explain_quicksort",
            "success": has_nlogn,
            "mentions_worst_case": has_n2,
            "elapsed": round(elapsed, 2)
        })
    else:
        tests.append({"test": "explain_quicksort", "success": False, "error": err[:100]})

    return tests

def test_factual_questions():
    """Test factual accuracy with verifiable answers."""
    print("\n" + "="*60)
    print("TEST 4: Factual Questions")
    print("="*60)

    questions = [
        ("What is the speed of light in meters per second?", "299792458", "speed of light"),
        ("What is the chemical formula for water?", "H2O", "water formula"),
        ("Who wrote Romeo and Juliet?", "Shakespeare", "shakespeare"),
        ("What is the largest planet in our solar system?", "Jupiter", "largest planet"),
        ("What year did World War II end?", "1945", "WWII end"),
    ]

    results = []
    for question, expected, test_name in questions:
        content, _, _, elapsed, err = chat([{
            "role": "user",
            "content": f"{question} Answer briefly."
        }], max_tokens=100, temperature=0.1)

        if not err:
            correct = expected.lower() in content.lower()
            print(f"  {test_name}: {'CORRECT' if correct else 'INCORRECT'} ({elapsed:.2f}s)")
            if not correct:
                print(f"    Expected '{expected}', got: {content[:80]}...")
            results.append({"test": test_name, "correct": correct, "elapsed": round(elapsed, 2)})
        else:
            print(f"  {test_name}: ERROR")
            results.append({"test": test_name, "correct": False, "error": err[:100]})

    accuracy = sum(1 for r in results if r.get("correct", False)) / len(results) * 100
    print(f"\n  Accuracy: {accuracy:.0f}%")
    return {"questions": results, "accuracy": accuracy}

def test_multimodal():
    """Check if model supports image input."""
    print("\n" + "="*60)
    print("TEST 5: Multimodal (Image) Support")
    print("="*60)

    # First check model info for vision capabilities
    try:
        resp = requests.get("http://localhost:31245/model_info")
        model_info = resp.json()
        print(f"  Model info: {json.dumps(model_info, indent=2)[:500]}")
    except Exception as e:
        print(f"  Could not get model info: {e}")

    # Try sending an image (a simple 1x1 red PNG)
    # This is a minimal valid PNG
    red_pixel_png = base64.b64encode(bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
        0x54, 0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x01, 0x00, 0x05, 0xFE,
        0xD4, 0xEF, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45,  # IEND chunk
        0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ])).decode()

    # Try OpenAI vision format
    payload = {
        "model": "Kimi-K2.5",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red_pixel_png}"}}
            ]
        }],
        "max_tokens": 50
    }

    print("\n  Attempting image input (OpenAI vision format)...")
    resp = requests.post(API_URL, headers=HEADERS, json=payload)

    result = {"supports_images": False}

    if resp.status_code == 200:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        print(f"  SUCCESS! Model accepted image input.")
        print(f"  Response: {content}")
        result["supports_images"] = True
        result["response"] = content
    else:
        error = resp.text[:300]
        print(f"  Image input not supported or failed.")
        print(f"  Error: {error}")
        result["error"] = error

        # Check if it's a format issue vs capability issue
        if "image" in error.lower() or "multimodal" in error.lower() or "vision" in error.lower():
            result["reason"] = "Model does not support vision/image input"
        else:
            result["reason"] = "Unknown error"

    return result

def test_reasoning():
    """Test reasoning capabilities."""
    print("\n" + "="*60)
    print("TEST 6: Reasoning Tasks")
    print("="*60)

    results = []

    # Math word problem
    print("\n  6a. Math Word Problem")
    content, _, comp_tok, elapsed, err = chat([{
        "role": "user",
        "content": "A train travels at 60 mph for 2 hours, then at 80 mph for 1.5 hours. What is the total distance traveled? Show your work."
    }], max_tokens=200, temperature=0.1)

    if not err:
        # Correct answer is 60*2 + 80*1.5 = 120 + 120 = 240 miles
        correct = "240" in content
        print(f"    Answer contains 240 miles: {correct} ({elapsed:.2f}s)")
        results.append({"test": "math_word_problem", "correct": correct, "elapsed": round(elapsed, 2)})
    else:
        results.append({"test": "math_word_problem", "correct": False, "error": err[:100]})

    # Logic puzzle
    print("\n  6b. Logic Puzzle")
    content, _, comp_tok, elapsed, err = chat([{
        "role": "user",
        "content": "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
    }], max_tokens=200, temperature=0.1)

    if not err:
        # The correct answer is NO - this is a logical fallacy
        correct = "no" in content.lower() or "cannot" in content.lower() or "not necessarily" in content.lower()
        print(f"    Correctly identifies we cannot conclude this: {correct} ({elapsed:.2f}s)")
        results.append({"test": "logic_puzzle", "correct": correct, "elapsed": round(elapsed, 2)})
    else:
        results.append({"test": "logic_puzzle", "correct": False, "error": err[:100]})

    return results

def main():
    print("="*60)
    print("KIMI-K2.5 COMPREHENSIVE TEST SUITE")
    print("="*60)
    print(f"Endpoint: {API_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Run all tests
    all_results["tokens_per_second"] = measure_tokens_per_second(runs=5)
    all_results["large_context"] = test_large_context()
    all_results["coding"] = test_coding_tasks()
    all_results["factual"] = test_factual_questions()
    all_results["multimodal"] = test_multimodal()
    all_results["reasoning"] = test_reasoning()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all_results["tokens_per_second"]:
        print(f"  Tokens/sec: {all_results['tokens_per_second']['average_tps']} avg")

    if all_results["large_context"]:
        print(f"  Large context ({all_results['large_context'].get('prompt_tokens', '?')} tokens): {'OK' if all_results['large_context']['success'] else 'FAILED'}")

    coding_pass = sum(1 for t in all_results["coding"] if t.get("success", False))
    print(f"  Coding tests: {coding_pass}/{len(all_results['coding'])} passed")

    print(f"  Factual accuracy: {all_results['factual']['accuracy']:.0f}%")

    print(f"  Image support: {'YES' if all_results['multimodal']['supports_images'] else 'NO'}")

    reasoning_pass = sum(1 for t in all_results["reasoning"] if t.get("correct", False))
    print(f"  Reasoning tests: {reasoning_pass}/{len(all_results['reasoning'])} passed")

    # Save results
    with open("/Users/sverker/repos/kimi/test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to test_results.json")

    return all_results

if __name__ == "__main__":
    main()
