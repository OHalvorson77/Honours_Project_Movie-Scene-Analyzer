import { useState, useRef, useCallback } from 'react';
import { Film, Clock, Zap, AlertTriangle, ArrowLeft } from 'lucide-react';
import { EmotionTimeline } from './components/EmotionTimeline';
import { TranscriptPanel } from './components/TranscriptPanel';
import { AnalysisPanel } from './components/AnalysisPanel';
import { UploadPage } from './components/UploadPage';
import { ProcessingPage } from './components/ProcessingPage';
import type { TranscriptSegment, SceneAnalysis } from './types';
import { EMOTION_COLORS } from './types';

type AppState = 'upload' | 'processing' | 'results';

function App() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [analysis, setAnalysis] = useState<SceneAnalysis | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Handle job started from upload
  const handleJobStarted = (newJobId: string) => {
    setJobId(newJobId);
    setAppState('processing');
    setError(null);
  };

  // Handle processing complete
  const handleProcessingComplete = useCallback(async () => {
    if (!jobId) return;

    try {
      const response = await fetch(`http://localhost:8000/api/results/${jobId}`);
      if (!response.ok) throw new Error('Failed to fetch results');

      const data = await response.json();
      setSegments(data.paired_data || []);
      setAnalysis(data.scene_analysis);
      setAppState('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load results');
      setAppState('upload');
    }
  }, [jobId]);

  // Handle processing error
  const handleProcessingError = (errorMsg: string) => {
    setError(errorMsg);
    setAppState('upload');
  };

  // Handle going back to upload
  const handleBackToUpload = () => {
    setAppState('upload');
    setJobId(null);
    setSegments([]);
    setAnalysis(null);
    setCurrentTime(0);
    setDuration(0);
  };

  // Find current segment based on video time
  const currentSegment = segments.find(
    (seg) => currentTime >= seg.start && currentTime < seg.end
  );

  const currentEmotion = currentSegment?.fused_emotion || 'neutral';

  // Handle video time updates
  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  // Handle video metadata loaded
  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  // Handle seeking
  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Calculate stats
  const conflictCount = segments.filter((s) => s.conflict_type).length;
  const uniqueEmotions = new Set(segments.map((s) => s.fused_emotion)).size;

  // Render based on app state
  if (appState === 'upload') {
    return (
      <div className="app">
        {error && (
          <div className="global-error">
            <AlertTriangle size={16} />
            <span>{error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}
        <UploadPage onJobStarted={handleJobStarted} />
      </div>
    );
  }

  if (appState === 'processing' && jobId) {
    return (
      <div className="app">
        <ProcessingPage
          jobId={jobId}
          onComplete={handleProcessingComplete}
          onError={handleProcessingError}
        />
      </div>
    );
  }

  // Results view
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <button className="back-btn" onClick={handleBackToUpload}>
              <ArrowLeft size={20} />
            </button>
            <div className="logo-icon">
              <Film size={20} />
            </div>
            <h1>Intelligent Scene Analyzer</h1>
          </div>
          <div className="header-stats">
            <div className="stat">
              <div className="stat-value">{segments.length}</div>
              <div className="stat-label">Segments</div>
            </div>
            <div className="stat">
              <div className="stat-value">{uniqueEmotions}</div>
              <div className="stat-label">Emotions</div>
            </div>
            <div className="stat">
              <div className="stat-value">{conflictCount}</div>
              <div className="stat-label">Conflicts</div>
            </div>
            <div className="stat">
              <div className="stat-value">{formatTime(duration)}</div>
              <div className="stat-label">Duration</div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        {/* Video Section */}
        <div className="video-section">
          {/* Video Player */}
          <div className="video-container">
            <video
              ref={videoRef}
              className="video-player"
              src={jobId ? `http://localhost:8000/api/video/${jobId}` : '/data/scene.mp4'}
              controls
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
            />
            <div className="video-overlay">
              <div className="current-emotion">
                <span
                  className="emotion-indicator"
                  style={{ backgroundColor: EMOTION_COLORS[currentEmotion] }}
                />
                <span className="emotion-label">{currentEmotion}</span>
                {currentSegment?.conflict_type && (
                  <AlertTriangle
                    size={14}
                    style={{ color: EMOTION_COLORS.fear, marginLeft: '0.5rem' }}
                  />
                )}
              </div>
              <span className="timestamp">
                <Clock size={14} style={{ marginRight: '0.375rem' }} />
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>
          </div>

          {/* Emotion Timeline */}
          <EmotionTimeline
            segments={segments}
            currentTime={currentTime}
            onSeek={handleSeek}
            duration={duration}
          />

          {/* Transcript */}
          <TranscriptPanel
            segments={segments}
            currentTime={currentTime}
            onSeek={handleSeek}
          />
        </div>

        {/* Sidebar */}
        <aside className="sidebar">
          {/* Summary Card */}
          <div className="summary-card">
            <h3>
              <Zap size={16} style={{ marginRight: '0.5rem' }} />
              Scene Summary
            </h3>
            <p>{analysis?.summary || 'No summary available.'}</p>
          </div>

          {/* Analysis Panel */}
          <AnalysisPanel analysis={analysis} />
        </aside>
      </main>
    </div>
  );
}

export default App;
