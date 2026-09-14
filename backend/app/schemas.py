from pydantic import BaseModel, Field


class AgentMessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    machine_id: str | None = None


class ResumeIn(BaseModel):
    approved: bool


class DashboardActionIn(BaseModel):
    thread_id: str | None = None
    target_version: str | None = None
