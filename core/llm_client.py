"""LLM client module for managing local LM Studio integration.

Centralizes all interactions with the local LM Studio model through
an OpenAI-compatible API. Handles configuration, retries, timeout,
and response formatting.

Integrates with LangChain for seamless LLM chain composition.
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from langchain_community.llms import OpenAI as LangChainOpenAI
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    text: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    latency_ms: Optional[float] = None


class LMStudioClient:
    """Client for interacting with LM Studio via OpenAI-compatible API.

    Provides retry logic, error handling, and logging for all LLM operations.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize the LM Studio client.

        Args:
            base_url: Base URL of the LM Studio API endpoint (e.g., http://localhost:1234/v1)
            model_name: Name of the model to use (e.g., gemma-4-e4b)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Base delay in seconds between retries (exponential backoff)
        """
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(api_key="not-needed", base_url=base_url)
        logger.info(
            f"Initialized LM Studio client: base_url={base_url}, "
            f"model={model_name}, timeout={timeout}s"
        )

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic.

        Args:
            func: Callable to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            APIError: If all retries are exhausted
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (APIConnectionError, RateLimitError, APIError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retries exhausted.")
        raise last_error

    def complete(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: Optional[int] = 512,
        stop: Optional[list[str]] = None,
    ) -> LLMResponse:
        """Generate text completion for a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter (note: not passed to API, kept for compatibility)
            max_tokens: Maximum tokens to generate, or None to omit the cap
            stop: Optional list of stop sequences to terminate generation

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            APIError: If the request fails after retries
        """
        start_time = time.time()

        def _call_api():
            request_kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "timeout": self.timeout,
            }
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            if stop is not None:
                request_kwargs["stop"] = stop
            return self.client.completions.create(**request_kwargs)

        try:
            response = self._execute_with_retry(_call_api)
            latency_ms = (time.time() - start_time) * 1000

            generated_text = response.choices[0].text
            logger.debug(
                f"Generated completion: model={self.model_name}, "
                f"prompt_len={len(prompt)}, response_len={len(generated_text)}, "
                f"latency={latency_ms:.0f}ms"
            )

            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                tokens_used=response.usage.completion_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}")
            raise

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Generate chat completion for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            temperature: Sampling temperature (0.0-2.0)
            top_p: Nucleus sampling parameter (0.0-1.0)
            top_k: Top-k sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated text and metadata

        Raises:
            APIError: If the request fails after retries
        """
        start_time = time.time()

        def _call_api():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )

        try:
            response = self._execute_with_retry(_call_api)
            latency_ms = (time.time() - start_time) * 1000

            generated_text = response.choices[0].message.content
            logger.debug(
                f"Generated chat completion: model={self.model_name}, "
                f"num_messages={len(messages)}, response_len={len(generated_text)}, "
                f"latency={latency_ms:.0f}ms"
            )

            return LLMResponse(
                text=generated_text,
                model=self.model_name,
                tokens_used=response.usage.completion_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Failed to generate chat completion: {str(e)}")
            raise

    def health_check(self) -> bool:
        """Check if the LM Studio API is accessible and responsive.

        Returns:
            True if the API is accessible, False otherwise
        """
        try:
            models = self.client.models.list()
            logger.info(f"Health check passed. Available models: {len(models.data)}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False


class LangChainCallbackHandler(BaseCallbackHandler):
    """Callback handler for logging LangChain LLM operations."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Called before LLM is invoked."""
        logger.debug(f"LangChain LLM start: prompts={len(prompts)}")

    def on_llm_end(self, response, **kwargs):
        """Called after LLM completes."""
        if response.generations:
            logger.debug(f"LangChain LLM end: generated {len(response.generations)} responses")

    def on_llm_error(self, error, **kwargs):
        """Called on LLM error."""
        logger.error(f"LangChain LLM error: {str(error)}")


def get_llm_client(
    base_url: str,
    model_name: str,
    timeout: int = 60,
    max_retries: int = 3,
) -> LMStudioClient:
    """Initialize and return an LM Studio LLM client.

    Args:
        base_url: Base URL of the LM Studio API endpoint
        model_name: Name of the model to use
        timeout: Request timeout in seconds
        max_retries: Maximum number of retries on failure

    Returns:
        Configured LMStudioClient instance
    """
    return LMStudioClient(
        base_url=base_url,
        model_name=model_name,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_langchain_llm(
    base_url: str,
    model_name: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> LangChainOpenAI:
    """Initialize and return a LangChain OpenAI wrapper for LM Studio.

    This wrapper is useful for LangChain chains, agents, and other integrations.

    Args:
        base_url: Base URL of the LM Studio API endpoint
        model_name: Name of the model to use
        temperature: Default sampling temperature
        max_tokens: Default maximum tokens to generate

    Returns:
        Configured LangChain OpenAI instance
    """
    logger.info(f"Initializing LangChain LLM: model={model_name}")

    llm = LangChainOpenAI(
        api_key="not-needed",
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[LangChainCallbackHandler()],
    )

    return llm


def query_llm(
    client: LMStudioClient,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """Send a prompt to the LLM and get a response.

    Convenience function for simple prompts.

    Args:
        client: LMStudioClient instance
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    response = client.complete(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.text
