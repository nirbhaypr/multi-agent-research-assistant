from collections.abc import Callable

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .models import TokenUsage


def invoke_structured[T: BaseModel](
    *,
    llm: ChatOpenAI,
    schema: type[T],
    messages: list[BaseMessage],
    record_usage: Callable[[TokenUsage | None], None] | None = None,
) -> T:
    structured_llm = llm.with_structured_output(
        schema,
        method="json_schema",
        strict=True,
        include_raw=True,
    )

    try:
        response = structured_llm.invoke(messages)
        reported_usage = response["raw"].usage_metadata
        usage = None

        if reported_usage is not None:
            usage = TokenUsage(
                input_tokens=reported_usage["input_tokens"],
                output_tokens=reported_usage["output_tokens"],
                total_tokens=reported_usage["total_tokens"],
            )
    except Exception:
        if record_usage is not None:
            record_usage(None)
        raise

    if record_usage is not None:
        record_usage(usage)

    metadata = response["raw"].response_metadata

    if metadata.get("status") not in (None, "completed"):
        raise RuntimeError("Model response was not completed")

    if metadata.get("finish_reason") in ("length", "content_filter"):
        raise RuntimeError("Model response was not completed")

    if response["parsing_error"] is not None:
        raise RuntimeError(
            "Model output could not be parsed"
        ) from response["parsing_error"]

    parsed = response["parsed"]

    if parsed is None:
        raise RuntimeError("Model returned no parsed output")

    return parsed
