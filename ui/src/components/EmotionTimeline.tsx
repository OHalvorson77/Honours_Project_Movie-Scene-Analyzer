import { useMemo } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { TranscriptSegment } from '../types';
import { EMOTION_COLORS } from '../types';

interface EmotionTimelineProps {
  segments: TranscriptSegment[];
  currentTime: number;
  onSeek: (time: number) => void;
  duration: number;
}

const EMOTIONS = ['angry', 'happy', 'sad', 'fear', 'neutral', 'disgust', 'surprise'];

export function EmotionTimeline({ segments, currentTime, onSeek, duration }: EmotionTimelineProps) {
  const chartData = useMemo(() => {
    const data: Array<{ time: number; [key: string]: number }> = [];
    
    // Create data points at each segment boundary
    segments.forEach((segment) => {
      const dist = segment.face_distribution;
      if (Object.keys(dist).length > 0) {
        data.push({
          time: segment.start,
          angry: dist.angry || 0,
          happy: dist.happy || 0,
          sad: dist.sad || 0,
          fear: dist.fear || 0,
          neutral: dist.neutral || 0,
          disgust: dist.disgust || 0,
          surprise: dist.surprise || 0,
        });
      }
    });
    
    return data;
  }, [segments]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleClick = (data: any) => {
    if (data?.activePayload?.[0]) {
      onSeek(data.activePayload[0].payload.time);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="timeline-container">
      <div className="timeline-header">
        <span className="timeline-title">Emotion Timeline</span>
        <div className="timeline-legend">
          {EMOTIONS.slice(0, 5).map((emotion) => (
            <div key={emotion} className="legend-item">
              <span 
                className="legend-dot" 
                style={{ backgroundColor: EMOTION_COLORS[emotion] }}
              />
              <span>{emotion}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="timeline-chart">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            onClick={handleClick}
            margin={{ top: 5, right: 5, left: -20, bottom: 5 }}
          >
            <defs>
              {EMOTIONS.map((emotion) => (
                <linearGradient
                  key={emotion}
                  id={`gradient-${emotion}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="0%"
                    stopColor={EMOTION_COLORS[emotion]}
                    stopOpacity={0.6}
                  />
                  <stop
                    offset="100%"
                    stopColor={EMOTION_COLORS[emotion]}
                    stopOpacity={0.1}
                  />
                </linearGradient>
              ))}
            </defs>
            <XAxis
              dataKey="time"
              tickFormatter={formatTime}
              tick={{ fontSize: 10, fill: '#606070' }}
              axisLine={{ stroke: '#2a2a3a' }}
              tickLine={{ stroke: '#2a2a3a' }}
              domain={[0, duration]}
            />
            <YAxis
              tick={{ fontSize: 10, fill: '#606070' }}
              axisLine={{ stroke: '#2a2a3a' }}
              tickLine={{ stroke: '#2a2a3a' }}
              domain={[0, 1]}
              ticks={[0, 0.5, 1]}
            />
            <Tooltip
              contentStyle={{
                background: '#15151f',
                border: '1px solid #2a2a3a',
                borderRadius: '8px',
                fontSize: '12px',
              }}
              labelFormatter={(value) => `Time: ${formatTime(value as number)}`}
              formatter={(value, name) => [
                `${((value as number) * 100).toFixed(1)}%`,
                (name as string).charAt(0).toUpperCase() + (name as string).slice(1),
              ]}
            />
            <ReferenceLine
              x={currentTime}
              stroke="#6366f1"
              strokeWidth={2}
              strokeDasharray="3 3"
            />
            {EMOTIONS.map((emotion) => (
              <Area
                key={emotion}
                type="monotone"
                dataKey={emotion}
                stackId="1"
                stroke={EMOTION_COLORS[emotion]}
                fill={`url(#gradient-${emotion})`}
                strokeWidth={1.5}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

