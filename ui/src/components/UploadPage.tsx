import { useState, useRef, useCallback } from 'react';
import { Upload, Film, Clock, AlertCircle, CheckCircle, Loader2 } from 'lucide-react';

interface UploadPageProps {
  onJobStarted: (jobId: string) => void;
}

export function UploadPage({ onJobStarted }: UploadPageProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && isValidVideo(droppedFile)) {
      setFile(droppedFile);
    } else {
      setError('Please upload a valid video file (MP4, MOV, AVI, MKV)');
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];
    if (selectedFile && isValidVideo(selectedFile)) {
      setFile(selectedFile);
    } else {
      setError('Please upload a valid video file (MP4, MOV, AVI, MKV)');
    }
  };

  const isValidVideo = (file: File) => {
    const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
    const validExtensions = ['.mp4', '.mov', '.avi', '.mkv'];
    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    return validTypes.includes(file.type) || validExtensions.includes(extension);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      onJobStarted(data.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <div className="upload-header">
          <div className="upload-logo">
            <Film size={32} />
          </div>
          <h1>Owen's Scene Analyzer</h1>
          <p>Upload a movie clip to analyze emotions, dialogue, and cinematic elements</p>
        </div>

        <div
          className={`upload-dropzone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
          />

          {file ? (
            <div className="file-preview">
              <CheckCircle size={48} className="success-icon" />
              <div className="file-info">
                <span className="file-name">{file.name}</span>
                <span className="file-size">{formatFileSize(file.size)}</span>
              </div>
              <button 
                className="change-file-btn"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                }}
              >
                Change file
              </button>
            </div>
          ) : (
            <>
              <Upload size={48} className="upload-icon" />
              <p className="upload-text">
                <span className="upload-cta">Click to upload</span> or drag and drop
              </p>
              <p className="upload-hint">MP4, MOV, AVI, MKV • Max 5 minutes</p>
            </>
          )}
        </div>

        {error && (
          <div className="upload-error">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        <button
          className="analyze-btn"
          onClick={handleUpload}
          disabled={!file || uploading}
        >
          {uploading ? (
            <>
              <Loader2 size={20} className="spin" />
              Uploading...
            </>
          ) : (
            <>
              <Film size={20} />
              Analyze Scene
            </>
          )}
        </button>

        <div className="upload-features">
          <div className="feature">
            <Clock size={20} />
            <div>
              <h4>Fast Processing</h4>
              <p>Analysis typically takes 2-5 minutes</p>
            </div>
          </div>
          <div className="feature">
            <Film size={20} />
            <div>
              <h4>Multimodal Analysis</h4>
              <p>Speech, facial expressions & scene understanding</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}



