from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, model_validator

from agentic_rl_forge.contracts.base import ContractModel, utc_now


class AuditSeverity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


class AuditFinding(ContractModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_.-]+$")
    severity: AuditSeverity
    message: str = Field(min_length=1)
    evidence: tuple[str, ...] = ()


class ReleaseAuditReport(ContractModel):
    project_path: str
    created_at: datetime = Field(default_factory=utc_now)
    ready: bool
    findings: tuple[AuditFinding, ...]
    pass_count: int = Field(ge=0)
    warning_count: int = Field(ge=0)
    error_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_summary(self) -> ReleaseAuditReport:
        counts = {
            severity: sum(item.severity is severity for item in self.findings)
            for severity in AuditSeverity
        }
        if self.pass_count != counts[AuditSeverity.PASS]:
            raise ValueError("pass_count does not match findings")
        if self.warning_count != counts[AuditSeverity.WARNING]:
            raise ValueError("warning_count does not match findings")
        if self.error_count != counts[AuditSeverity.ERROR]:
            raise ValueError("error_count does not match findings")
        if self.ready != (self.error_count == 0):
            raise ValueError("ready must reflect error_count")
        return self
