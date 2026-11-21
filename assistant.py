import os
import torch

from transformers import pipeline
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

# -----------------------------
# 1. REVIEW SUMMARIZATION
# -----------------------------

_summarizer = None


def load_summarizer():
    global _summarizer

    if _summarizer is None:
        device = 0 if torch.cuda.is_available() else -1

        _summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            framework="pt",
            device=device
        )

    return _summarizer


def summarize_reviews(reviews, chunk_size=3):

    if not reviews:
        raise ValueError("Reviews cannot be empty.")

    summarizer = load_summarizer()

    chunks = [
        reviews[i:i + chunk_size]
        for i in range(0, len(reviews), chunk_size)
    ]

    chunk_summaries = []

    for chunk in chunks:
        result = summarizer(
            " ".join(chunk),
            max_length=80,
            min_length=20,
            do_sample=False,
            truncation=True
        )

        chunk_summaries.append(
            result[0]["summary_text"]
        )

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    final_result = summarizer(
        " ".join(chunk_summaries),
        max_length=100,
        min_length=25,
        do_sample=False,
        truncation=True
    )

    return final_result[0]["summary_text"]


# -----------------------------
# 2. PRODUCT SUPPORT CHATBOT
# -----------------------------

MODEL_ID = "gemini-3.5-flash-lite"

def create_product_chatbot(
    product_name,
    product_metadata,
    product_reviews
):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("Gemini API key not found.")

    client = genai.Client(api_key=api_key)

    system_instruction = f"""
    You are RetailMind, an intelligent product support assistant.

    PRODUCT INFORMATION:
    Name: {product_name}
    Metadata: {product_metadata}

    CUSTOMER REVIEWS:
    {product_reviews}

    INSTRUCTIONS:
    1. Answer using the supplied product information.
    2. Identify positive and negative feedback.
    3. Never invent product specifications.
    4. Distinguish facts from customer opinions.
    5. If information is unavailable, say so.
    6. Maintain a professional tone.
    """

    chat = client.chats.create(
        model=MODEL_ID,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        )
    )

    return client, chat