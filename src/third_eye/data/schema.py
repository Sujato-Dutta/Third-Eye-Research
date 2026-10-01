from dataclasses import asdict, dataclass, field

from third_eye.io import digest

ROLES = ("train", "target_dev", "ood_dev", "retention_dev")
TEST_ROLES = ("target_test", "ood_test", "retention_test")


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    answer: str
    split: str
    task: str = "math"
    difficulty: str = "default"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id or not self.prompt.strip() or not self.answer.strip():
            raise ValueError("Examples require id, prompt, and answer")
        if self.split not in ROLES + TEST_ROLES:
            raise ValueError(
                "Final test data needs an explicit *_test role; unknown split"
            )
        if self.task not in {"math", "code", "exact"}:
            raise ValueError("Unknown task")

    @property
    def prompt_hash(self):
        return digest(" ".join(self.prompt.split()).casefold())


@dataclass(frozen=True)
class Correction:
    example: Example
    completion: str
    attempt: int

    def __post_init__(self):
        if self.example.split != "train" or not self.completion.strip():
            raise ValueError(
                "Only nonempty verified training completions may form updates"
            )

    def to_dict(self):
        return {
            "example": asdict(self.example),
            "completion": self.completion,
            "attempt": self.attempt,
        }
