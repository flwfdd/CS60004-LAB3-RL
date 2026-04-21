from dataclasses import dataclass


@dataclass
class DPOSample:
    prompt: str
    chosen: str
    rejected: str
