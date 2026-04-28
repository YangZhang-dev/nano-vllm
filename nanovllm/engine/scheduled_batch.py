from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

from nanovllm.engine.sequence import Sequence


ScheduledKind = Literal["prefill", "decode"]


@dataclass(slots=True)
class ScheduledItem:
    seq: Sequence
    num_scheduled_tokens: int
    kind: ScheduledKind


@dataclass(slots=True)
class ScheduledBatch:
    items: list[ScheduledItem]

    @classmethod
    def from_legacy(cls, seqs: list[Sequence], is_prefill: bool) -> "ScheduledBatch":
        kind: ScheduledKind = "prefill" if is_prefill else "decode"
        return cls([
            ScheduledItem(seq, seq.num_scheduled_tokens, kind)
            for seq in seqs
        ])

    @property
    def seqs(self) -> list[Sequence]:
        return [item.seq for item in self.items]

    @property
    def num_seqs(self) -> int:
        return len(self.items)

    @property
    def prefill_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.items if item.kind == "prefill")

    @property
    def decode_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.items if item.kind == "decode")

    @property
    def num_scheduled_tokens(self) -> int:
        return self.prefill_tokens + self.decode_tokens

    @property
    def is_prefill(self) -> bool:
        return self.prefill_tokens > 0 and self.decode_tokens == 0

    @property
    def is_decode(self) -> bool:
        return self.decode_tokens > 0 and self.prefill_tokens == 0

    @property
    def is_mixed(self) -> bool:
        return self.prefill_tokens > 0 and self.decode_tokens > 0

    @property
    def legacy_token_count(self) -> int:
        if self.is_decode:
            return -self.decode_tokens
        return self.num_scheduled_tokens


@dataclass(slots=True)
class StepMetrics:
    num_scheduled_tokens: int
    prefill_tokens: int
    decode_tokens: int
    num_seqs: int
    is_prefill: bool
    is_mixed: bool
    preemptions: int = 0

    @classmethod
    def from_batch(cls, batch: ScheduledBatch, preemptions: int = 0) -> "StepMetrics":
        return cls(
            num_scheduled_tokens=batch.num_scheduled_tokens,
            prefill_tokens=batch.prefill_tokens,
            decode_tokens=batch.decode_tokens,
            num_seqs=batch.num_seqs,
            is_prefill=batch.is_prefill,
            is_mixed=batch.is_mixed,
            preemptions=preemptions,
        )

    @property
    def legacy_token_count(self) -> int:
        if self.decode_tokens and not self.prefill_tokens:
            return -self.decode_tokens
        return self.num_scheduled_tokens


@dataclass(slots=True)
class StepOutput:
    outputs: list[tuple[int, list[int]]]
    metrics: StepMetrics
    scheduled_batch: ScheduledBatch

    @property
    def num_tokens(self) -> int:
        return self.metrics.legacy_token_count

    @property
    def seqs(self) -> list[Sequence]:
        return self.scheduled_batch.seqs

    def __iter__(self) -> Iterator[object]:
        yield self.outputs
        yield self.num_tokens
        yield self.seqs
