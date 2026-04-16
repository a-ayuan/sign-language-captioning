import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import RecordPanel from "./components/RecordPanel";
import ResultsPanel from "./components/ResultsPanel";

export default function App() {
  const [mode, setMode] = useState("upload");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const resetResults = () => {
    setResult(null);
    setError("");
  };

  const runInference = async (blobOrFile, filename) => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", blobOrFile, filename);

      const response = await fetch("/api/predict-video", {
        method: "POST",
        body: formData,
      });

      const rawText = await response.text();
      let data = null;

      try {
        data = rawText ? JSON.parse(rawText) : null;
      } catch {
        throw new Error(`Server returned invalid JSON: ${rawText || "empty response"}`);
      }

      if (!response.ok) {
        throw new Error(data?.detail || "Inference failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>Sign Language Captioning</h1>
          <p>Upload a video or record yourself to generate streaming-style captions.</p>
        </div>
      </header>

      <div className="tab-row">
        <button
          className={mode === "upload" ? "tab active" : "tab"}
          onClick={() => setMode("upload")}
        >
          Upload Video
        </button>
        <button
          className={mode === "record" ? "tab active" : "tab"}
          onClick={() => setMode("record")}
        >
          Record Yourself
        </button>
      </div>

      <main className="main-grid">
        <section className="panel left-panel">
          {mode === "upload" ? (
            <UploadPanel onSubmit={runInference} loading={loading} />
          ) : (
            <RecordPanel
              onSubmit={runInference}
              onResetResults={resetResults}
              loading={loading}
            />
          )}

          {error ? <div className="error-box">{error}</div> : null}
        </section>

        <section className="panel right-panel">
          <ResultsPanel result={result} loading={loading} />
        </section>
      </main>
    </div>
  );
}