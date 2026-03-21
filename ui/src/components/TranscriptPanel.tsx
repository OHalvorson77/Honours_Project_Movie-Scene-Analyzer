import { useEffect, useRef } from 'react';
import { AlertTriangle } from 'lucide-react';
import type { TranscriptSegment } from '../types';
import { EMOTION_COLORS } from '../types';

interface TranscriptPanelProps {
  segments: TranscriptSegment[];
  currentTime: number;
  onSeek: (time: number) => void;
}

export function TranscriptPanel({ segments, currentTime, onSeek }: TranscriptPanelProps) {
  const activeRef = useRef<HTMLDivElement>(null);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getActiveSegmentIndex = () => {
    return segments.findIndex(
      (seg) => currentTime >= seg.start && currentTime < seg.end
    );
  };

  const activeIndex = getActiveSegmentIndex();

  // Auto-scroll to active segment
  useEffect(() => {
    if (activeRef.current) {
      activeRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [activeIndex]);

  return (
    <div className="transcript-panel">
      <div className="panel-header">
        <h3 className="panel-title">Transcript</h3>
      </div>
      <div className="transcript-list">
        {segments.map((segment, index) => (
          <div
            key={index}
            ref={index === activeIndex ? activeRef : null}
            className={`transcript-segment ${index === activeIndex ? 'active' : ''}`}
            onClick={() => onSeek(segment.start)}
          >
            <div className="segment-header">
              <span className="segment-time">
                {formatTime(segment.start)} - {formatTime(segment.end)}
              </span>
              <div className="segment-emotions">
                {segment.speech_emotion && segment.speech_emotion !== 'unknown' && (
                  <span
                    className="emotion-badge speech"
                    style={{
                      backgroundColor: `${EMOTION_COLORS[segment.speech_emotion]}20`,
                      color: EMOTION_COLORS[segment.speech_emotion],
                    }}
                  >
                    🎤 {segment.speech_emotion}
                  </span>
                )}
                {segment.face_emotion && (
                  <span
                    className="emotion-badge face"
                    style={{
                      backgroundColor: `${EMOTION_COLORS[segment.face_emotion]}20`,
                      color: EMOTION_COLORS[segment.face_emotion],
                    }}
                  >
                    😊 {segment.face_emotion}
                  </span>
                )}
              </div>
            </div>
            <p className="segment-text">{segment.text}</p>
            {segment.conflict_type && (
              <div className="conflict-indicator">
                <AlertTriangle size={12} />
                <span>
                  {segment.conflict_type === 'emotion_mismatch'
                    ? 'Audio/visual mismatch'
                    : segment.conflict_type.replace(/_/g, ' ')}
                </span>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

