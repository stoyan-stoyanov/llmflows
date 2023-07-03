# pylint: disable=R1710

"""Module containing helper functions for the LLM classes."""

import time
import asyncio
import logging
from openai.error import (
    APIError,
    Timeout,
    RateLimitError,
    APIConnectionError,
    InvalidRequestError,
    AuthenticationError,
    ServiceUnavailableError,
)


def call_with_retry(api_obj, max_retries, *args, **kwargs):
    """Repeatedly invokes the provided api object up to the specified maximum number
    of retries.

    Args:
        asynchronous (bool): flag specifying wether to use the async version of the API
        api_obj (obj): The API object to be invoked.
        max_retries (int): The maximum number of retry attempts.
        *args: Variable length argument list for the API object.
        **kwargs: Arbitrary keyword arguments for the API object.

    Returns:
        tuple: The response from the API object and the number of retries.

    Raises:
        APIError: If there is an issue on the provider's side.
        Timeout: If the request timed out.
        RateLimitError: If the rate limit has been hit.
        APIConnectionError: If there's an issue connecting to the services.
        InvalidRequestError: If the request was malformed or missing some required
        parameters.
        AuthenticationError: If the API key or token was invalid, expired, or revoked.
        ServiceUnavailableError: If there's an issue on the server's side.
    """
    max_delay = 10
    delay_multiplier = 1.5
    num_retries = 0
    delay = 1

    while num_retries <= max_retries:
        try:
            response = api_obj.create(*args, **kwargs)
            return response, num_retries

        except (
            APIError,
            Timeout,
            RateLimitError,
            APIConnectionError,
            ServiceUnavailableError,
        ) as error:
            num_retries += 1
            logging.warning("Retrying: Attempt %s. Error: %s", num_retries, str(error))
            time.sleep(min(delay, max_delay))
            delay *= delay_multiplier

        except (InvalidRequestError, AuthenticationError) as error:
            logging.error(
                "An error occurred that cannot be resolved by retrying. Error: %s",
                str(error),
            )
            raise


async def async_call_with_retry(api_obj, max_retries, *args, **kwargs):
    """
    Async implementation of a function that repeatedly invokes the provided api object 
    up to the specified maximum number of retries in an asynchronous manner.

    Args:
        api_obj (obj): The API object to be invoked.
        max_retries (int): The maximum number of retry attempts.
        *args: Variable length argument list for the API object.
        **kwargs: Arbitrary keyword arguments for the API object.

    Returns:
        tuple: The response from the API object and the number of retries.

    Raises:
        APIError: If there is an issue on the provider's side.
        Timeout: If the request timed out.
        RateLimitError: If the rate limit has been hit.
        APIConnectionError: If there's an issue connecting to the services.
        InvalidRequestError: If the request was malformed or missing some required
        parameters.
        AuthenticationError: If the API key or token was invalid, expired, or revoked.
        ServiceUnavailableError: If there's an issue on the server's side.
    """
    max_delay = 10
    delay_multiplier = 1.5
    num_retries = 0
    delay = 1

    while num_retries <= max_retries:
        try:
            response = await api_obj.acreate(*args, **kwargs)
            return response, num_retries

        except (
            APIError,
            Timeout,
            RateLimitError,
            APIConnectionError,
            ServiceUnavailableError,
        ) as error:
            num_retries += 1
            logging.warning("Retrying: Attempt %s. Error: %s", num_retries, str(error))
            await asyncio.sleep(min(delay, max_delay))
            delay *= delay_multiplier

        except (InvalidRequestError, AuthenticationError) as error:
            logging.error(
                "An error occurred that cannot be resolved by retrying. Error: %s",
                str(error),
            )
            raise
