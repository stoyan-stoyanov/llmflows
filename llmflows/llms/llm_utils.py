# pylint: disable=R1710

"""Module containing helper functions for the LLM classes."""

import time
import asyncio
import logging


def call_with_retry(func, exceptions_to_retry, max_retries, *args, **kwargs):
    """
    Repeatedly invokes the provided function up to the specified maximum number of 
    retries.

    Args:
        func: The function to be invoked.
        exceptions_to_retry: A tuple of exception types to retry on.
        max_retries: The maximum number of retry attempts.
        *args: Variable length argument list for the function.
        **kwargs: Arbitrary keyword arguments for the function.

    Returns:
        A tuple containing the response from the function and the number of retries.

    Raises:
        Immediately raises any exceptions not in exceptions_to_retry. If the maximum
        number of retries is reached, raises an Exception containing all of the
        exceptions encountered during the retries.
    """
    max_delay = 10
    delay_multiplier = 1.5
    num_retries = 0
    delay = 1
    exceptions_encountered = []

    while num_retries <= max_retries:
        try:
            response = func(*args, **kwargs)
            return response, num_retries

        except exceptions_to_retry as error:
            num_retries += 1
            exceptions_encountered.append(error)
            logging.warning("Retrying: Attempt %s. Error: %s", num_retries, str(error))
            time.sleep(min(delay, max_delay))
            delay *= delay_multiplier

        except Exception as error:  # pylint: disable=broad-exception-raised
            logging.error(
                "An error occurred that cannot be resolved by retrying. Error: %s",
                str(error),
            )
            raise

    if exceptions_encountered:
        error_messages = "\n".join(str(e) for e in exceptions_encountered)
        raise Exception(
            f"All retries exhausted. Encountered exceptions:\n{error_messages}"
        )


async def async_call_with_retry(
    async_func, exceptions_to_retry, max_retries, *args, **kwargs
):
    """
    Repeatedly invokes the provided async function up to the specified maximum number 
    of retries.

    Args:
        async_func: The async function to be awaited.
        exceptions_to_retry: A tuple of exception types to retry on.
        max_retries: The maximum number of retry attempts.
        *args: Variable length argument list for the function.
        **kwargs: Arbitrary keyword arguments for the function.

    Returns:
        A tuple containing the response from the function and the number of retries.

    Raises:
        Immediately raises any exceptions not in exceptions_to_retry. If the maximum
        number of retries is reached, raises an Exception containing all of the
        exceptions encountered during the retries.
    """
    max_delay = 10
    delay_multiplier = 1.5
    num_retries = 0
    delay = 1
    exceptions_encountered = []

    while num_retries <= max_retries:
        try:
            response = await async_func(*args, **kwargs)
            return response, num_retries

        except exceptions_to_retry as error:
            num_retries += 1
            exceptions_encountered.append(error)
            logging.warning("Retrying: Attempt %s. Error: %s", num_retries, str(error))
            await asyncio.sleep(min(delay, max_delay))
            delay *= delay_multiplier

        except Exception as error:  # pylint: disable=broad-exception-raised
            logging.error(
                "An error occurred that cannot be resolved by retrying. Error: %s",
                str(error),
            )
            raise

    if exceptions_encountered:
        error_messages = "\n".join(str(e) for e in exceptions_encountered)
        raise Exception(
            f"All retries exhausted. Encountered exceptions:\n{error_messages}"
        )
