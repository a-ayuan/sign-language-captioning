import { useMemo, useState } from "react";

export default function UploadPanel({ onSubmit, loading }) {
  const [file, setFile] = useState(null);
  const previewUrl = useMemo(() => (file ? URL.createObjectURL(file) : ""), [file]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;
    await onSubmit(file, file.name);
  };

  return (
    <div>
      <h2>Upload a Video</h2>
      <p className="muted">Supported formats: MP4, MOV, AVI, MKV, WEBM</p>

      <form onSubmit={handleSubmit} className="stack">
        <label className="file-drop">
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,video/webm"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <span>{file ? file.name : "Choose a video file"}</span>
        </label>

                {previewUrl ? (
                  <video className="video-preview" src={previewUrl} controls />
                ) : (
                  <div className="video-placeholder">Video preview appears here</div>
                )}

                <button className="primary-btn" type="submit" disabled={!file || loading}>
                  {loading ? "Running Inference..." : "Generate Caption"}
                </button>
              </form>
            </div>
          );
        }