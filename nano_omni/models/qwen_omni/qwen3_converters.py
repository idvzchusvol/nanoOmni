from __future__ import annotations

from nano_omni.types import SamplingParams, StageInput, StageOutput

_DEFAULT_TALKER_SAMPLING = SamplingParams(
    temperature=0.9,
    top_k=50,
    max_tokens=4096,
    stop_token_ids=[2150],
    repetition_penalty=1.05,
)

_DEFAULT_NUM_CODEBOOKS = 8


def qwen3_thinker2talker(
    thinker_out: StageOutput,
    talker_sampling: SamplingParams | None = None,
) -> StageInput:
    """
    Convert Thinker StageOutput → Talker StageInput.

    - embeddings: Thinker's last hidden state, used as Talker's prefix embedding
    - sampling_params: Talker-specific codec generation params
    """
    if thinker_out.embeddings is None:
        raise ValueError(
            f"thinker2talker: request {thinker_out.request_id} missing embeddings. "
            "Ensure ARStageEngine has output_hidden_states=True."
        )
    return StageInput(
        request_id=thinker_out.request_id,
        token_ids=thinker_out.token_ids,
        embeddings=thinker_out.embeddings,
        sampling_params=talker_sampling or _DEFAULT_TALKER_SAMPLING,
        extra={"thinker_token_ids": thinker_out.token_ids},
    )


def qwen3_talker2code2wav(
    talker_out: StageOutput,
    num_codebooks: int = _DEFAULT_NUM_CODEBOOKS,
) -> StageInput:
    """
    Convert Talker StageOutput → Code2Wav StageInput.

    - token_ids: codec code token_ids (num_codebooks * T total)
    - extra["num_codebooks"]: used by CodecStageEngine to reshape
    """
    return StageInput(
        request_id=talker_out.request_id,
        token_ids=talker_out.token_ids,
        extra={"num_codebooks": num_codebooks},
    )
