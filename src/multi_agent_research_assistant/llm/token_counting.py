from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import OpenAI, pydantic_function_tool
from pydantic import BaseModel


def count_structured_input(
    *,
    client: OpenAI,
    model: str,
    schema: type[BaseModel],
    messages: list[BaseMessage],
) -> int:
    """Count our stateless text requests with reasoning_effort='none'."""
    items = []

    for message in messages:
        if type(message) not in (SystemMessage, HumanMessage):
            raise ValueError("Only SystemMessage and HumanMessage are supported")

        if not isinstance(message.content, str):
            raise ValueError("Only plain text content is supported")

        if message.name is not None or message.additional_kwargs:
            raise ValueError("Message names and additional kwargs are unsupported")

        items.append(
            {
                "type": "message",
                "role": ("system" if isinstance(message, SystemMessage) else "user"),
                "content": message.content,
            }
        )

    strict_schema = pydantic_function_tool(schema)["function"]["parameters"]

    response = client.responses.input_tokens.count(
        model=model,
        input=items,
        reasoning={"effort": "none"},
        text={
            "format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": strict_schema,
                "strict": True,
            },
        },
    )

    count = response.input_tokens
    if type(count) is not int or count < 0:
        raise ValueError("Input token count must be a nonnegative integer")

    return count
