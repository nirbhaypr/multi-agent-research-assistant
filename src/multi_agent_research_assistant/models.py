from pydantic import BaseModel, Field, ConfigDict, AwareDatetime, HttpUrl

class SubQuestion(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    completion_criteria: str = Field(min_length=1)


class ResearchPlan(BaseModel):
    model_config = ConfigDict(
            extra="forbid",
            str_strip_whitespace=True,
        )

    question: str = Field(min_length=1)
    subquestions: list[SubQuestion] = Field(min_length=1)


class Finding(BaseModel):
    model_config = ConfigDict(
                extra="forbid",
                str_strip_whitespace=True,
            )

    id: str = Field(min_length=1)
    subquestion_id: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    source_url: HttpUrl
    snippet: str = Field(min_length=1)
    retrieved_at: AwareDatetime


class ReportClaim(BaseModel):
    model_config = ConfigDict(
                extra="forbid",
                str_strip_whitespace=True,
            )

    text: str = Field(min_length=1)
    finding_ids: list[str] = Field(min_length=1)


class ResearchReport(BaseModel):
    model_config = ConfigDict(
                extra="forbid",
                str_strip_whitespace=True,
            )

    title: str = Field(min_length=1)
    claims: list[ReportClaim] = Field(min_length=1)


class TokenUsage(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)