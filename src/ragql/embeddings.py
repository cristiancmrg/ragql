# SPDX-License-Identifier: MPL-2.0
"""
ragql.embeddings
~~~~~~~~~~~~~~~~
*  Generates embeddings via Ollama **or** OpenAI.
*  Provides a thin wrapper to call an LLM chat endpoint for final answer generation.
*  All behaviour is driven by `Settings` so the rest of the library never
   reads env-vars directly.
"""

from __future__ import annotations
from typing import Iterable, List
import logging

import numpy as np
import requests

from .config import Settings

logger = logging.getLogger(__name__)


# Public helper – get_embeddings
def get_embeddings(texts: List[str], cfg: Settings) -> np.ndarray:
    """Return an array of embedding vectors for the given texts.

    Chooses the embedding backend based on `cfg.embed_provider`:
    - "ollama": calls the Ollama embeddings endpoint
    - "openai": calls the OpenAI embeddings API

    Args:
        texts (List[str]): A list of strings to embed.
        cfg (Settings): Configuration object, must have `embed_provider`.

    Returns:
        np.ndarray: Float32 array of shape (len(texts), embedding_dim).

    Raises:
        ValueError: If `cfg.embed_provider` is not "ollama" or "openai".
    """

    count = len(texts)
    provider = cfg.embed_provider
    logger.info("get_embeddings: %d texts via provider '%s'", count, provider)

    if provider == "ollama":
        vecs = _ollama_embed(texts, cfg)
    elif provider == "openai":
        vecs = _openai_embed(texts, cfg)
    else:
        logger.error("Invalid embed provider: %r", provider)
        raise ValueError(
            f"Embed provider: {provider} is not valid;"
            "it should be either (ollama or openai)"
        )

    logger.debug("Embeddings shape: %r", getattr(vecs, "shape", None))

    return vecs


# Public helpers – chat completion:
def call_ollama_chat(prompt: str, context: str, cfg: Settings) -> str:
    """Generate a chat response using an Ollama model.

    Sends a formatted prompt and context to the Ollama `/api/generate` endpoint
    and returns the model’s reply as a stripped string.

    Args:
        prompt (str): The user’s query or instruction.
        context (str): Supporting context to include in the prompt.
        cfg (Settings): Configuration object containing `ollama_url`.

    Returns:
        str: The model’s text response with leading/trailing whitespace removed.

    Raises:
        requests.HTTPError: If the HTTP request returns a non-2xx status code.
        RuntimeError: If the JSON response does not include a "response" field.
    """

    logger.info("call_ollama_chat: sending request to %s", cfg.ollama_url)

    payload = {
        "model": "mistral:7b-instruct",
        "prompt": _format_prompt(prompt, context),
        "stream": False,
        "options": {"temperature": 0},
    }

    r = requests.post(f"{cfg.ollama_url}/api/generate", json=payload, timeout=120)

    logger.debug("Ollama generate status: %d", r.status_code)

    r.raise_for_status()
    js = r.json()

    logger.debug("Ollama generate response JSON: %r", js)

    if "response" not in js:
        logger.error("Ollama chat error: %r", js)
        raise RuntimeError(f"Ollama chat error → {js.get('error', js)}")

    response = js["response"].strip()
    logger.info("Ollama chat completed: response length %d", len(response))
    return response


def call_openai_chat(prompt: str, context: str, cfg: Settings) -> str:
    """Generate a chat response using the OpenAI API.

    Uses the `openai` Python package to call the chat completion endpoint
    with a formatted prompt and context, returning the assistant’s reply.

    Args:
        prompt (str): The user’s query or instruction.
        context (str): Supporting context to include in the prompt.
        cfg (Settings): Configuration object containing `openai_key`.

    Returns:
        str: The model’s text response with leading/trailing whitespace removed.

    Raises:
        RuntimeError: If the `openai` package is not installed.
        openai.error.OpenAIError: If the API call fails for any reason.
        RuntimeError: If the API returns no choices or an empty response.
    """

    logger.info("call_openai_chat: starting OpenAI completion")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        logger.error("OpenAI package not installed")
        raise RuntimeError(
            "`openai` package not installed: pip install openai"
        ) from exc

    client = OpenAI(api_key=cfg.openai_key)
    logger.debug("OpenAI client created with key length %d", len(cfg.openai_key))

    rs = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": _format_prompt(prompt, context)}],
        temperature=0,
        max_tokens=256,
    )

    logger.debug("OpenAI raw response: %r", rs)

    # Extract the content of the first choice
    try:
        content = rs.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        logger.error("OpenAI chat returned no choices: %r", rs)
        raise RuntimeError("OpenAI chat returned no response choices") from exc

    if not content:
        logger.error("OpenAI chat returned empty content")
        raise RuntimeError("OpenAI chat returned empty content")

    result = content.strip()
    logger.info("OpenAI chat completed: response length %d", len(result))
    return result


# Internal helpers:
def _ollama_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    """Generate embeddings for each text prompt using an Ollama server.

    Sends each prompt in `texts` individually to the Ollama embeddings endpoint
    specified by `cfg.ollama_url` and returns a NumPy array of embedding vectors.

    Args:
        texts (Iterable[str]): A sequence of text strings to embed.
        cfg (Settings): Configuration object containing `ollama_url`
            and the selected `embed_model_name`.

    Returns:
        np.ndarray: An array of shape (len(texts), embedding_dim) with dtype float32.

    Raises:
        requests.HTTPError: If the Ollama API returns a non-2xx status code.
        RuntimeError: If the response JSON does not contain an "embedding" field.
    """

    model = cfg.embed_model_name
    logger.info(
        "_ollama_embed: sending %d prompts to %s/embeddings with model %s",
        len(list(texts)),
        cfg.ollama_url,
        model,
    )

    vecs = []
    for prompt in texts:  # Ollama v0.1.x only supports single-prompt payloads
        r = requests.post(
            f"{cfg.ollama_url}/api/embeddings",
            json={"model": model, "prompt": prompt},
            timeout=60,
        )

        logger.debug("Ollama embed status for prompt[...]: %d", r.status_code)

        r.raise_for_status()
        js = r.json()

        logger.debug("Ollama embed JSON: %r", js)

        if "embedding" not in js:
            logger.error("Ollama embed error → %r", js)
            raise RuntimeError(f"Ollama embed error → {js.get('error', js)}")

        vecs.append(js["embedding"])

    arr = np.array(vecs, dtype="float32")
    logger.debug("_ollama_embed returning array of shape %r", arr.shape)
    return arr


def _openai_embed(texts: Iterable[str], cfg: Settings) -> np.ndarray:
    """Generate embeddings for a batch of texts using the OpenAI API.

    Uses the `openai` Python package to call the embeddings endpoint with
    the model specified in `cfg.embed_model_name`.

    Args:
        texts (Iterable[str]): A sequence of text strings to embed.
        cfg (Settings): Configuration object containing `openai_key`
            and the selected `embed_model_name`.

    Returns:
        np.ndarray: An array of shape (len(texts), embedding_dim) with dtype float32.

    Raises:
        RuntimeError: If the `openai` package is not installed.
        openai.error.OpenAIError: If the API call fails for any reason.
    """

    logger.info(
        "_openai_embed: creating embeddings for %d texts with model %s",
        len(list(texts)),
        cfg.embed_model_name,
    )

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        logger.error("OpenAI package not installed")
        raise RuntimeError(
            "`openai` package not installed: pip install openai"
        ) from exc

    client = OpenAI(api_key=cfg.openai_key)
    logger.debug("OpenAI embed client created")

    res = client.embeddings.create(
        model=cfg.embed_model_name,
        input=list(texts),
    )

    embeddings = np.array([d.embedding for d in res.data], dtype="float32")
    logger.debug("_openai_embed returning array of shape %r", embeddings.shape)
    return embeddings


def _format_prompt(question: str, context: str) -> str:
    """Build and log a prompt for LogGPT using the provided context.

    This helper constructs a prompt that instructs LogGPT to answer the
    given `question` using *only* the supplied `context`. It then prints
    the prompt (for debugging purposes) before returning it.

    Args:
        question (str): The user’s question to be answered.
        context (str): Relevant context that the model may reference.

    Returns:
        str: The fully formatted prompt string ready for the LLM.
    """
    logger.debug(
        "_format_prompt: question length %d, context length %d",
        len(question),
        len(context),
    )

    prompt = (
        "You are LogGPT. Using *only* the context below, answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    logger.debug("Formatted prompt: %s", prompt)

    return prompt
