from strenum import StrEnum


class Status(StrEnum):
    PASS = "completed"
    FAIL = "failed"
    NOT_STARTED = "not started"
