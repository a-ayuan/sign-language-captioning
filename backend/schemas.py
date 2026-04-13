from pydantic import BaseModel


class ChunkPrediction(BaseModel):
    chunk_index: int
    start_frame: int
    end_frame: int
    decoded_tokens: list[str]
    committed_tokens: list[str]


class InferenceResponse(BaseModel):
    final_caption: str
    chunks: list[ChunkPrediction]
    elapsed_seconds: float
    num_chunks: int
    caption_churn: float
    video_filename: str