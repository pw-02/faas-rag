import os
import json
import nltk
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import concurrent
from concurrent.futures import ThreadPoolExecutor
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple
# from nltk.tokenize import sent_tokenize
# nltk.download("punkt")

# ----------------------- Custom Headers -----------------------
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/58.0.3029.110 Safari/537.36",
    "Referer": "https://www.google.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.
    """
    try:
        full_text = full_text[:50000]

        snippet_norm = remove_punctuation(snippet.lower())
        snippet_words = set(snippet_norm.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = sent_tokenize(full_text)
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        for sentence in sentences:
            key_sentence = remove_punctuation(sentence.lower())
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context

        return False, full_text[:context_chars * 2]

    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_pdf_text(url: str) -> str:
    """
    Extract text from a PDF.
    """
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"

        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        cleaned_text = " ".join(full_text.split()[:600])
        return cleaned_text

    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


def extract_text_from_url(url: str, use_jina: bool = False, jina_api_key: Optional[str] = None,
                         snippet: Optional[str] = None) -> str:
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.
    """
    try:
        if use_jina:
            if not jina_api_key:
                return "Error: use_jina=True but no jina_api_key provided."

            jina_headers = {
                "Authorization": f"Bearer {jina_api_key}",
                "X-Return-Format": "markdown",
            }
            response_text = requests.get(f"https://r.jina.ai/{url}", headers=jina_headers, timeout=20).text

            # Remove URLs in markdown parentheses/brackets
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response_text).replace("---", "-").replace("===", "=").replace("   ", " ")

        else:
            response = session.get(url, timeout=20)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            if "pdf" in content_type.lower():
                return extract_pdf_text(url)

            try:
                soup = BeautifulSoup(response.text, "lxml")
            except Exception:
                soup = BeautifulSoup(response.text, "html.parser")

            text = soup.get_text(separator=" ", strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            return context if success else text

        return text[:8000]

    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def fetch_page_content(urls, max_workers: int = 32, use_jina: bool = False,
                       jina_api_key: Optional[str] = None, snippets: Optional[dict] = None) -> dict:
    """
    Concurrently fetch content from multiple URLs.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                extract_text_from_url,
                url,
                use_jina,
                jina_api_key,
                snippets.get(url) if snippets else None
            ): url
            for url in urls
        }

        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            # time.sleep(0.2)  # Simple rate limiting

    return results


# ----------------------- Brave Search -----------------------
def brave_web_search(
    query: str,
    api_key: str,
    endpoint: str = "https://api.search.brave.com/res/v1/web/search",
    country: str = "US",
    search_lang: str = "en",
    count: int = 10,
    offset: int = 0,
    safesearch: str = "moderate",  # "off" | "moderate" | "strict"
    freshness: Optional[str] = None,
    timeout: int = 20
) -> dict:
    """
    Perform a web search using the Brave Search API.
    """
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "country": country,
        "search_lang": search_lang,
        "count": count,
        "offset": offset,
        "safesearch": safesearch,
    }
    if freshness:
        params["freshness"] = freshness

    try:
        resp = requests.get(endpoint, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Timeout:
        print(f"Brave Search request timed out ({timeout} seconds) for query: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Brave Search request: {e}")
        try:
            print("Response body:", resp.text[:500])
        except Exception:
            pass
        return {}


def extract_relevant_info(search_results: dict) -> list:
    """
    Extract relevant information from Brave search results.
    """
    useful_info = []

    web = search_results.get("web", {})
    results = web.get("results", []) or []

    for idx, r in enumerate(results):
        title = r.get("title", "") or r.get("name", "")
        url = r.get("url", "")
        snippet = r.get("description", "") or r.get("snippet", "") or ""

        site_name = ""
        profile = r.get("profile")
        if isinstance(profile, dict):
            site_name = profile.get("name", "")

        date = ""
        if isinstance(r.get("page_age"), str):
            date = r["page_age"]
        elif isinstance(r.get("age"), str):
            date = r["age"]

        useful_info.append({
            "id": idx + 1,
            "title": title,
            "url": url,
            "site_name": site_name,
            "date": date,
            "snippet": snippet,
            "context": ""
        })

    return useful_info


# ------------------------------------------------------------
if __name__ == "__main__":
    query = "Structure of dimethyl fumarate"

    BRAVE_API_KEY = "BSAwFH9sF7CRI00LnmfXoFTXuY-ZXUg"

    print("Performing Brave Web Search...")
    search_results = brave_web_search(query, BRAVE_API_KEY, count=1)

    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info["url"], use_jina=False, jina_api_key=os.environ.get("JINA_API_KEY"))
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info["snippet"])
            if success:
                info["context"] = context
            else:
                info["context"] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info["context"] = f"Failed to fetch full text: {full_text}"

    # Print result JSON if you want
    print(json.dumps(extracted_info, indent=2, ensure_ascii=False))