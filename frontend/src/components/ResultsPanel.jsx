export default function ResultsPanel({ result, loading }) {
  if (loading) {
    return (
      <div>
        <h2>Results</h2>
        <div className="loading-box">Processing video and generating captions...</div>
      </div>
    );
  }

  if (!result) {
    return (
      <div>
        <h2>Results</h2>
        <div className="empty-box">Inference results will appear here.</div>
      </div>
    );
  }

  return (
      <div>
        <h2>Results</h2>

        <div className="result-card">
          <div className="result-label">Video</div>
          <div className="result-value">{result.video_filename}</div>
        </div>

        <div className="result-card highlight">
          <div className="result-label">Final Caption</div>
          <div className="caption-text">{result.final_caption || "No caption produced"}</div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <span>Elapsed</span>
            <strong>{result.elapsed_seconds.toFixed(3)}s</strong>
          </div>
          <div className="stat-card">
            <span>Chunks</span>
            <strong>{result.num_chunks}</strong>
          </div>
          <div className="stat-card">
            <span>Caption Churn</span>
            <strong>{result.caption_churn.toFixed(3)}</strong>
                    </div>
                  </div>

                  <div className="chunk-section">
                    <h3>Chunk Predictions</h3>
                    <div className="chunk-list">
                      {result.chunks.map((chunk) => (
                        <div key={chunk.chunk_index} className="chunk-card">
                          <div className="chunk-header">
                            <strong>Chunk {chunk.chunk_index}</strong>
                            <span>
                              Frames {chunk.start_frame} - {chunk.end_frame}
                            </span>
                          </div>
                          <div>
                            <span className="mini-label">Decoded:</span>{" "}
                            {chunk.decoded_tokens.join(" ") || "—"}
                          </div>
                          <div>
                            <span className="mini-label">Committed:</span>{" "}
                            {chunk.committed_tokens.join(" ") || "—"}
                          </div>
                        </div>
                      ))}
                  </div>
                        </div>
                      </div>
                    );
                  }