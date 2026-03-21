import { useState } from 'react';
import {
  Film,
  Users,
  Camera,
  TrendingUp,
  AlertTriangle,
  Lightbulb,
  ChevronDown,
} from 'lucide-react';
import type { SceneAnalysis } from '../types';
import { EMOTION_COLORS } from '../types';

interface AnalysisPanelProps {
  analysis: SceneAnalysis | null;
}

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

function Section({ title, icon, defaultOpen = false, children }: SectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="analysis-section">
      <div className="section-header" onClick={() => setIsOpen(!isOpen)}>
        <span className="section-title">
          <span className="section-icon">{icon}</span>
          {title}
        </span>
        <ChevronDown
          size={16}
          className={`section-toggle ${isOpen ? 'open' : ''}`}
        />
      </div>
      {isOpen && <div className="section-content">{children}</div>}
    </div>
  );
}

export function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  if (!analysis) {
    return (
      <div className="analysis-panel">
        <div className="panel-header">
          <h3 className="panel-title">Scene Analysis</h3>
        </div>
        <div className="section-content">
          <p>No analysis available. Run the pipeline first.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="analysis-panel">
      <div className="panel-header">
        <h3 className="panel-title">Scene Analysis</h3>
      </div>

      <Section title="Scene Overview" icon={<Film size={16} />} defaultOpen>
        <p>
          <strong>Setting:</strong> {analysis.scene_overview.setting}
        </p>
        <p>
          <strong>Time:</strong> {analysis.scene_overview.time_of_day}
        </p>
        <p>
          <strong>Atmosphere:</strong> {analysis.scene_overview.atmosphere}
        </p>
        <p>
          <strong>Mood:</strong> {analysis.scene_overview.mood}
        </p>
      </Section>

      <Section title="Characters" icon={<Users size={16} />}>
        {analysis.characters.map((char, i) => (
          <div key={i} style={{ marginBottom: '1rem' }}>
            <p>
              <strong>Description:</strong> {char.description}
            </p>
            <p>
              <strong>Emotional Journey:</strong> {char.emotional_journey}
            </p>
            <p>
              <strong>Performance:</strong> {char.performance_notes}
            </p>
          </div>
        ))}
      </Section>

      <Section title="Cinematic Elements" icon={<Camera size={16} />}>
        <p>
          <strong>Camera:</strong> {analysis.cinematic_elements.camera_work}
        </p>
        <p>
          <strong>Lighting:</strong> {analysis.cinematic_elements.lighting}
        </p>
        <p>
          <strong>Colors:</strong> {analysis.cinematic_elements.color_palette}
        </p>
        <p>
          <strong>Techniques:</strong>{' '}
          {analysis.cinematic_elements.notable_techniques}
        </p>
      </Section>

      <Section title="Emotional Arc" icon={<TrendingUp size={16} />}>
        <p>{analysis.emotional_arc.progression}</p>
        <div className="theme-tags">
          {analysis.emotional_arc.key_moments.map((moment, i) => (
            <span key={i} className="theme-tag">
              {moment}
            </span>
          ))}
        </div>
      </Section>

      <Section title="Emotion Conflicts" icon={<AlertTriangle size={16} />}>
        <div className="conflicts-list">
          {analysis.conflict_interpretations.map((conflict, i) => (
            <div key={i} className="conflict-card">
              <div className="conflict-time">{conflict.timestamp}</div>
              <div className="conflict-emotions">
                <span
                  style={{
                    color: EMOTION_COLORS[conflict.speech_emotion] || '#999',
                  }}
                >
                  🎤 {conflict.speech_emotion}
                </span>
                <span className="conflict-arrow">→</span>
                <span
                  style={{
                    color: EMOTION_COLORS[conflict.face_emotion] || '#999',
                  }}
                >
                  😊 {conflict.face_emotion}
                </span>
              </div>
              <p className="conflict-interpretation">{conflict.interpretation}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Themes" icon={<Lightbulb size={16} />} defaultOpen>
        <p>{analysis.themes.narrative_significance}</p>
        <div className="theme-tags">
          {analysis.themes.central_themes.map((theme, i) => (
            <span key={i} className="theme-tag">
              {theme}
            </span>
          ))}
        </div>
      </Section>
    </div>
  );
}

