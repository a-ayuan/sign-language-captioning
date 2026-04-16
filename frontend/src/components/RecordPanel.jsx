import { useEffect, useRef, useState } from "react";

export default function RecordPanel({ onSubmit, onResetResults, loading }) {
  const previewRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState("");

  useEffect(() => {
    return () => {
      cleanupRecorder();
      cleanupStream();
      cleanupRecordedUrl();
    };
  }, [recordedUrl]);

  const cleanupRecorder = () => {
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
    } catch (error) {
      console.error("Recorder cleanup failed:", error);
    }
    mediaRecorderRef.current = null;
  };

  const cleanupStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (previewRef.current) {
      previewRef.current.srcObject = null;
    }
    setIsCameraReady(false);
  };

  const cleanupRecordedUrl = () => {
    if (recordedUrl) {
      URL.revokeObjectURL(recordedUrl);
    }
  };

  const clearOldClip = () => {
    cleanupRecordedUrl();
    setRecordedBlob(null);
    setRecordedUrl("");
    chunksRef.current = [];

    if (onResetResults) {
      onResetResults();
    }
  };

  const hasLiveStream = () => {
    return !!(
      streamRef.current &&
      streamRef.current.getVideoTracks().length > 0 &&
      streamRef.current.getVideoTracks().some((track) => track.readyState === "live")
    );
  };

  const attachStreamToPreview = async (stream) => {
    if (!previewRef.current) return;
    previewRef.current.srcObject = stream;
    try {
      await previewRef.current.play();
    } catch (error) {
      console.error("Preview play failed:", error);
    }
  };

  const startCamera = async () => {
    try {
      if (hasLiveStream()) {
        await attachStreamToPreview(streamRef.current);
        setIsCameraReady(true);
        return;
      }

      cleanupStream();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      streamRef.current = stream;
      await attachStreamToPreview(stream);
      setIsCameraReady(true);
    } catch (error) {
      console.error("Could not start camera:", error);
      setIsCameraReady(false);
    }
  };

  const getSupportedRecorderOptions = () => {
    const candidates = [
      { mimeType: "video/webm;codecs=vp9" },
      { mimeType: "video/webm;codecs=vp8" },
      { mimeType: "video/webm" },
      { mimeType: "video/mp4" },
      undefined,
    ];

    for (const option of candidates) {
      if (!option) return undefined;
      if (window.MediaRecorder?.isTypeSupported?.(option.mimeType)) {
        return option;
      }
    }

    return undefined;
  };

  const beginRecording = () => {
    if (!hasLiveStream()) return;

    try {
      chunksRef.current = [];

      const options = getSupportedRecorderOptions();
      const recorder = options
        ? new MediaRecorder(streamRef.current, options)
        : new MediaRecorder(streamRef.current);

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const blobType = chunksRef.current[0]?.type || "video/webm";
        const blob = new Blob(chunksRef.current, { type: blobType });
        const url = URL.createObjectURL(blob);
        setRecordedBlob(blob);
        setRecordedUrl(url);
        setIsRecording(false);
      };

      recorder.onerror = (event) => {
        console.error("Recorder error:", event);
        setIsRecording(false);
      };

      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Could not start recording:", error);
      setIsRecording(false);
    }
  };

  const startRecording = async () => {
    if (isRecording || loading) return;

    try {
      if (!hasLiveStream()) {
        await startCamera();
      }

      if (!hasLiveStream()) return;

      cleanupRecorder();
      clearOldClip();

      // let React clear the old clip before starting again
      setTimeout(() => {
        beginRecording();
      }, 50);
    } catch (error) {
      console.error("Could not start recording:", error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (!mediaRecorderRef.current) return;
    if (mediaRecorderRef.current.state === "inactive") return;
    mediaRecorderRef.current.stop();
  };

  const submitRecording = async () => {
    if (!recordedBlob) return;
    await onSubmit(recordedBlob, "recording.webm");
  };

  return (
    <div>
      <h2>Record Yourself</h2>
      <p className="muted">Open your webcam, record a short clip, then caption it.</p>

      <div className="stack">
        <video
          ref={previewRef}
          className="video-preview"
          autoPlay
          muted
          playsInline
        />

        <div className="button-row">
          <button
            className="secondary-btn"
            onClick={startCamera}
            disabled={loading || isRecording}
            type="button"
          >
            {isCameraReady ? "Restart Camera" : "Start Camera"}
          </button>

          {!isRecording ? (
            <button
              className="primary-btn"
              onClick={startRecording}
              disabled={loading}
              type="button"
            >
              {recordedUrl ? "Record Again" : "Start Recording"}
            </button>
          ) : (
            <button
              className="danger-btn"
              onClick={stopRecording}
              type="button"
            >
              Stop Recording
            </button>
          )}
        </div>

        {recordedUrl ? (
          <>
            <h3>Recorded Clip</h3>
            <video className="video-preview" src={recordedUrl} controls />
            <button
              className="primary-btn"
              onClick={submitRecording}
              disabled={loading}
              type="button"
            >
              {loading ? "Running Inference..." : "Caption Recording"}
            </button>
          </>
        ) : null}
      </div>
    </div>
  );
}