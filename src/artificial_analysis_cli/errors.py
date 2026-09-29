"""Stable command failures shared by the CLI and its internal modules."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never


class ErrorCode(StrEnum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    API_ERROR = "API_ERROR"
    INVALID_RESPONSE = "INVALID_RESPONSE"

    @property
    def exit_code(self) -> int:
        match self:
            case ErrorCode.INVALID_ARGUMENT | ErrorCode.AUTH_REQUIRED | ErrorCode.UNAUTHORIZED:
                return 2
            case ErrorCode.FORBIDDEN:
                return 3
            case ErrorCode.RATE_LIMITED:
                return 4
            case ErrorCode.TIMEOUT | ErrorCode.NETWORK_ERROR | ErrorCode.API_ERROR | ErrorCode.INVALID_RESPONSE:
                return 1
            case _:
                assert_never(self)


class CliError(Exception):
    """A safe, actionable failure that can be shown to people or tools."""

    def __init__(self, code: ErrorCode, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint

    @property
    def exit_code(self) -> int:
        return self.code.exit_code
