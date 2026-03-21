// Types for the Movie Scene Analyzer UI

export interface Frame {
  filename: string;
  path: string;
  timestamp: number;
}

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  speech_emotion: string;
  speech_confidence: number;
  face_emotion: string | null;
  face_confidence: number;
  face_distribution: Record<string, number>;
  faces_detected: number;
  fused_emotion: string;
  fused_confidence: number;
  fused_source: string;
  emotions_match: boolean | null;
  conflict_type: string | null;
  frames: Frame[];
}

export interface SceneOverview {
  setting: string;
  time_of_day: string;
  atmosphere: string;
  mood: string;
}

export interface Character {
  description: string;
  emotional_journey: string;
  performance_notes: string;
}

export interface CinematicElements {
  camera_work: string;
  lighting: string;
  color_palette: string;
  notable_techniques: string;
}

export interface EmotionalArc {
  progression: string;
  key_moments: string[];
}

export interface ConflictInterpretation {
  timestamp: string;
  speech_emotion: string;
  face_emotion: string;
  interpretation: string;
  narrative_significance: string;
}

export interface Themes {
  central_themes: string[];
  narrative_significance: string;
}

export interface SceneAnalysis {
  scene_overview: SceneOverview;
  characters: Character[];
  cinematic_elements: CinematicElements;
  emotional_arc: EmotionalArc;
  conflict_interpretations: ConflictInterpretation[];
  themes: Themes;
  summary: string;
}

export type EmotionType = 
  | 'angry' 
  | 'calm' 
  | 'disgust' 
  | 'fear' 
  | 'happy' 
  | 'neutral' 
  | 'sad' 
  | 'surprise'
  | 'surprised'
  | 'unknown';

export const EMOTION_COLORS: Record<string, string> = {
  angry: '#ef4444',
  calm: '#22c55e',
  disgust: '#a855f7',
  fear: '#f97316',
  happy: '#facc15',
  neutral: '#6b7280',
  sad: '#3b82f6',
  surprise: '#ec4899',
  surprised: '#ec4899',
  unknown: '#9ca3af',
};



