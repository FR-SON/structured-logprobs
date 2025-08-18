import json
from collections.abc import Sequence
from functools import singledispatch
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.responses import ParsedResponse, ParsedResponseOutputMessage
from openai.types.responses.response_output_text import Logprob
from pydantic import BaseModel

from structured_logprobs.helpers import extract_json_data, extract_json_data_inline

MISSING_LOGPROBS_MESSAGE = "The 'logprobs' field is missing"

"""

This module provides utilities to work with OpenAI chat completion responses,
enhancing them by embedding log probabilities into the data.
The module contains a function for mapping characters to token indices (`map_characters_to_token_indices`) and two methods for incorporating log probabilities:
1. Adding log probabilities as a separate field in the response (`add_logprobs`).
2. Embedding log probabilities inline within the message content (`add_logprobs_inline`).

Classes:
    - ChatCompletionWithLogProbs: Represents a chat completion response with added log probabilities.

"""


class ResponseWithLogProbs(BaseModel):
    value: ChatCompletion | ParsedResponse
    log_probs: list[Any]


def map_characters_to_token_indices(extracted_data_token: Sequence[Logprob | ChatCompletionTokenLogprob]) -> list[int]:
    """
    Maps each character in the JSON string output to its corresponding token index.

    Args:
    extracted_data_token : A list of `TokenLogprob` objects, where each object represents a token and its associated data.

    Returns:
    A list of integers where each position corresponds to a character in the concatenated JSON string,
    and the integer at each position is the index of the token responsible for generating that specific character.
    Example:
        >>> tokens = [ChatCompletionTokenLogprob(token='{'),
                      ChatCompletionTokenLogprob(token='"key1"'),
                      ChatCompletionTokenLogprob(token=': '),
                      ChatCompletionTokenLogprob(token='"value1"'),
                      ChatCompletionTokenLogprob(token='}')]
        >>> map_characters_to_token_indices(tokens)
        [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4]
    """

    token_indices = []

    for token_idx, token_data in enumerate(extracted_data_token):
        token_text = token_data.token
        token_indices.extend([token_idx] * len(token_text))

    return token_indices


@singledispatch
def add_logprobs(response: Any) -> ResponseWithLogProbs:
    """
    Generic function to add log probabilities to a response.
    This base implementation raises an error for unsupported types.
    """
    raise NotImplementedError(f"Unsupported response type: {type(response).__name__}")


@add_logprobs.register(ChatCompletion)
def _(chat_completion_response: ChatCompletion) -> ResponseWithLogProbs:
    """
    Adds log probabilities to the chat completion response and returns a
    ResponseWithLogProbs object.

    Args:
        chat_completion_response: The OpenAI chat completion response.

    Returns:
        An object containing:
            - The original chat completion response.
            - A `log_probs` field, structured like the message.content of the response,
              where values are replaced with their respective log-probabilities.
    Raises:
        AttributeError: If any 'choice' in the response does not contain 'logprobs'.

    """

    logprobs_data = []
    for choice in chat_completion_response.choices:
        if hasattr(choice, "logprobs") and choice.logprobs and choice.logprobs.content:
            extracted_data = choice.message.content
            if extracted_data is None:
                continue
            logprobs_list = choice.logprobs.content
            token_indices = map_characters_to_token_indices(logprobs_list)
            json_dict = extract_json_data(extracted_data, logprobs_list, token_indices)
            logprobs_data.append(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    return ResponseWithLogProbs(value=chat_completion_response, log_probs=logprobs_data)


@add_logprobs.register(ParsedResponse)
def _(parsed_response: ParsedResponse) -> ResponseWithLogProbs:
    """
    Adds log probabilities to the response object and returns a
    ResponseWithLogProbs object.

    Args:
        parsed_response: The OpenAI chat completion response.

    Returns:
        An object containing:
            - The original parsed response.
            - A `log_probs` field, structured like the message.content of the response,
              where values are replaced with their respective log-probabilities.
    Raises:
        AttributeError: If any 'output' in the response does not contain 'logprobs'.

    """
    logprobs_data = []
    for output_message in parsed_response.output:
        if not isinstance(output_message, ParsedResponseOutputMessage) or output_message.content is None:
            continue

        for content_block in output_message.content:
            if hasattr(content_block, "text") and hasattr(content_block, "logprobs"):
                extracted_data = content_block.text
                logprobs_list = content_block.logprobs
                if not logprobs_list:
                    raise AttributeError(MISSING_LOGPROBS_MESSAGE)
                token_indices = map_characters_to_token_indices(logprobs_list)
                json_dict = extract_json_data(extracted_data, logprobs_list, token_indices)
                logprobs_data.append(json_dict)

    return ResponseWithLogProbs(value=parsed_response, log_probs=logprobs_data)


def add_logprobs_inline(chat_completion_response: ChatCompletion) -> ChatCompletion:
    """
    Embeds inline log probabilities into the content of the message in the chat completion response.
    This is only supported for the legacy ChatCompletion API, since the ParsedResponse contains a pared object
    for the content, that would be out of sync with the inline log probabilities.

    Args:
        chat_completion_response: The OpenAI chat completion response.

    Returns:
        ChatCompletion: The modified chat completion response object, where the content of the message
            is replaced with a dictionary that includes also inline log probabilities for atomic values.

    Raises:
        AttributeError: If the 'logprobs' field is not present in the response.
    """

    for choice in chat_completion_response.choices:
        # Check if the 'logprobs' field is present
        if hasattr(choice, "logprobs") and choice.logprobs is not None and choice.logprobs.content is not None:
            extracted_data = choice.message.content
            if extracted_data is None:
                continue
            logprobs_list = choice.logprobs.content
            token_indices = map_characters_to_token_indices(logprobs_list or [])
            json_dict = (
                extract_json_data_inline(extracted_data, logprobs_list or [], token_indices) if extracted_data else {}
            )
            choice.message.content = json.dumps(json_dict)
        else:
            raise AttributeError(MISSING_LOGPROBS_MESSAGE)

    return chat_completion_response


if __name__ == "__main__":  # pragma: no cover
    pass
