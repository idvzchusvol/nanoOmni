# tests/test_scheduler.py
import pytest
from nano_omni.types import StageInput, SamplingParams
from nano_omni.scheduler.scheduler import Scheduler, SequenceState, ScheduleBatch


def make_input(rid: str, length: int) -> StageInput:
    return StageInput(request_id=rid, token_ids=list(range(length)),
                      sampling_params=SamplingParams(max_tokens=10))


def test_empty_scheduler():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    assert sched.has_unfinished() is False


def test_add_request():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 5))
    assert sched.has_unfinished() is True


def test_short_prompt_fully_prefilled_in_one_step():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 4))  # prompt length 4 < chunk_size 8
    batch = sched.schedule()
    assert len(batch.prefill_seqs) == 1
    seq, chunk = batch.prefill_seqs[0]
    assert seq.inp.request_id == "r1"
    assert chunk == [0, 1, 2, 3]
    assert len(batch.decode_seqs) == 0
    # After full prefill, seq moves to running
    assert seq.prefill_offset == 4
    assert seq.is_prefilling is False


def test_long_prompt_chunked_across_steps():
    sched = Scheduler(max_batch_size=4, chunk_size=4, max_tokens_per_step=32)
    sched.add(make_input("r1", 10))   # needs 3 chunks: 4+4+2
    batch1 = sched.schedule()
    assert len(batch1.prefill_seqs) == 1
    _, chunk1 = batch1.prefill_seqs[0]
    assert chunk1 == [0, 1, 2, 3]   # first 4 tokens
    assert sched.has_unfinished() is True  # still prefilling
    batch2 = sched.schedule()
    _, chunk2 = batch2.prefill_seqs[0]
    assert chunk2 == [4, 5, 6, 7]
    batch3 = sched.schedule()
    _, chunk3 = batch3.prefill_seqs[0]
    assert chunk3 == [8, 9]


def test_running_requests_go_to_decode():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 3))  # short prompt, fully prefilled
    batch1 = sched.schedule()
    seq = batch1.prefill_seqs[0][0]
    # seq is now in running (is_prefilling=False)
    assert not seq.is_prefilling
    batch2 = sched.schedule()
    assert len(batch2.decode_seqs) == 1
    assert batch2.decode_seqs[0].inp.request_id == "r1"


def test_finish_removes_from_running():
    sched = Scheduler(max_batch_size=4, chunk_size=8, max_tokens_per_step=32)
    sched.add(make_input("r1", 3))
    sched.schedule()   # moves r1 to running
    sched.finish("r1")
    assert sched.has_unfinished() is False


def test_token_budget_limits_concurrent_prefill():
    # budget=6: decode takes 0 (no running), 2 prefills of 3 tokens each
    sched = Scheduler(max_batch_size=4, chunk_size=3, max_tokens_per_step=6)
    sched.add(make_input("r1", 3))
    sched.add(make_input("r2", 3))
    batch = sched.schedule()
    assert len(batch.prefill_seqs) == 2


def test_decode_budget_limits_new_prefill():
    # max_tokens_per_step=4, 3 running decode seqs consume 3 tokens → 1 left for prefill
    sched = Scheduler(max_batch_size=8, chunk_size=4, max_tokens_per_step=4)
    # put 3 seqs in running
    for rid in ["r1", "r2", "r3"]:
        sched.add(make_input(rid, 2))
        sched.schedule()   # each short, fully prefilled, now running
    sched.add(make_input("r4", 4))
    batch = sched.schedule()
    assert len(batch.decode_seqs) == 3
    # prefill chunk limited to 1 token (budget=4-3=1)
    if batch.prefill_seqs:
        _, chunk = batch.prefill_seqs[0]
        assert len(chunk) == 1
