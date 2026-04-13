import { useEffect, useRef, useState } from "react";

export default function RecordPanel({ onSubmit, loading }) {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);

  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedUrl, setRecordedUrl] = useState("");

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (recordedUrl) {
        URL.revokeObjectURL(recordedUrl);
      }
    };
  }, [recordedUrl]);

  const startCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraReady(true);
    };

    const startRecording = () => {
      if (!streamRef.current) return;

      chunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current, {
        mimeType: "video/webm",
      });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
            const blob = new Blob(chunksRef.current, { type: "video/webm" });
            const url = URL.createObjectURL(blob);
            setRecordedBlob(blob);
            setRecordedUrl(url);
          };

          mediaRecorderRef.current = recorder;
          recorder.start();
          setIsRecording(true);
        };

        const stopRecording = () => {
          mediaRecorderRef.current?.stop();
          setIsRecording(false);
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
                <video ref={videoRef} className="video-preview" autoPlay muted playsInline />

                <div className="button-row">
                  <button className="secondary-btn" onClick={startCamera} disabled={isCameraReady}>
                    {isCameraReady ? "Camera Ready" : "Start Camera"}
                  </button>

                  {!isRecording ? (
                    <button className="primary-btn" onClick={startRecording} disabled={!isCameraReady || loading}>
                      Start Recording
                    </button>
                  ) : (
                    <button className="danger-btn" onClick={stopRecording}>
                      Stop Recording
                    </button>
                  )}
                </div>

                {recordedUrl ? (
                          <>
                            <h3>Recorded Clip</h3>
                            <video className="video-preview" src={recordedUrl} controls />
                            <button className="primary-btn" onClick={submitRecording} disabled={loading}>
                              {loading ? "Running Inference..." : "Caption Recording"}
                            </button>
                          </>
                        ) : null}
                      </div>
                    </div>
                  );
                }
