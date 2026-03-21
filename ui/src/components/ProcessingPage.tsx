import { useState, useEffect } from 'react';
import { Loader2, CheckCircle, AlertCircle, Film } from 'lucide-react';

interface ProcessingPageProps {
  jobId: string;
  onComplete: () => void;
  onError: (error: string) => void;
}

interface JobStatus {
  job_id: string;
  status: string;
  progress: number;
  current_step: string;
  error: string | null;
}

const PIPELINE_STEPS = [
  { name: 'Transcribing audio', threshold: 20 },
  { name: 'Extracting frames', threshold: 35 },
  { name: 'Analyzing speech emotions', threshold: 50 },
  { name: 'Analyzing facial expressions', threshold: 70 },
  { name: 'Fusing multimodal data', threshold: 80 },
  { name: 'Selecting keyframes', threshold: 85 },
  { name: 'Generating AI analysis', threshold: 100 },
];

export function ProcessingPage({ jobId, onComplete, onError }: ProcessingPageProps) {
  const [status, setStatus] = useState<JobStatus | null>(null);

  useEffect(() => {
    const pollStatus = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/status/${jobId}`);
        if (!response.ok) throw new Error('Failed to fetch status');
        
        const data: JobStatus = await response.json();
        setStatus(data);

        if (data.status === 'completed') {
          onComplete();
        } else if (data.status === 'failed') {
          onError(data.error || 'Processing failed');
        }
      } catch (err) {
        console.error('Status poll error:', err);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 1000);
    return () => clearInterval(interval);
  }, [jobId, onComplete, onError]);

  const getStepStatus = (stepThreshold: number) => {
    if (!status) return 'pending';
    if (status.progress >= stepThreshold) return 'completed';
    if (status.progress >= stepThreshold - 15) return 'active';
    return 'pending';
  };

  return (
    <div className="processing-page">
      <div className="processing-container">
        <div className="processing-header">
          <div className="processing-icon">
            <Film size={32} />
          </div>
          <h1>Analyzing Your Scene</h1>
          <p>{status?.current_step || 'Initializing...'}</p>
        </div>

        <div className="progress-container">
          <div className="progress-bar">
            <div 
              className="progress-fill"
              style={{ width: `${status?.progress || 0}%` }}
            />
          </div>
          <span className="progress-text">{status?.progress || 0}%</span>
        </div>

        <div className="pipeline-steps">
          {PIPELINE_STEPS.map((step, index) => {
            const stepStatus = getStepStatus(step.threshold);
            return (
              <div key={index} className={`pipeline-step ${stepStatus}`}>
                <div className="step-indicator">
                  {stepStatus === 'completed' ? (
                    <CheckCircle size={20} />
                  ) : stepStatus === 'active' ? (
                    <Loader2 size={20} className="spin" />
                  ) : (
                    <div className="step-dot" />
                  )}
                </div>
                <span className="step-name">{step.name}</span>
              </div>
            );
          })}
        </div>

        {status?.status === 'failed' && (
          <div className="processing-error">
            <AlertCircle size={20} />
            <span>{status.error}</span>
          </div>
        )}
      </div>
    </div>
  );
}



