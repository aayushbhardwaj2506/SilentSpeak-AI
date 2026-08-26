import { useEffect, useRef, useState } from 'react'
import { GESTURE_VOCABULARY } from './lib/GestureVocabulary'
import { HandLandmarker, FaceLandmarker, PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision'
import { GestureRecognizer } from './lib/GestureRecognizer'
import { ActionObserver } from './lib/ActionObserver'
import type { GestureResult } from './lib/GestureRecognizer'
import type { ActionObservationPayload } from './lib/ActionObserver'
import './App.css'

interface AgentDecision {
  intent: string;
  decision: string;
  response_text: string;
  confidence: number;
}

interface HandTrackingData {
  numHands: number;
  handednesses: string[];
  distanceNormalized: number | null;
}

interface PerceptionData {
  hands: {
    landmarks: any[][];
    handednesses: string[];
  };
  face: {
    landmarks: any[][];
  };
  pose: {
    landmarks: any[][];
  };
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const orbRef = useRef<HTMLDivElement>(null);
  const smoothedDistanceRef = useRef<number>(0.5);
  const smoothedPosXRef = useRef<number>(50);
  const smoothedPosYRef = useRef<number>(50);
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | null>(null);
  
  const [gestureResult, setGestureResult] = useState<GestureResult | null>(null);
  const [agentDecision, setAgentDecision] = useState<AgentDecision | null>(null);
  const [handTrackingData, setHandTrackingData] = useState<HandTrackingData>({ numHands: 0, handednesses: [], distanceNormalized: null });
  
  const actionObserverRef = useRef(new ActionObserver());
  const [actionPayload, setActionPayload] = useState<ActionObservationPayload | null>(null);

  // Use a ref for real-time perception data to avoid 60fps re-renders
  const perceptionDataRef = useRef<PerceptionData>({
    hands: { landmarks: [], handednesses: [] },
    face: { landmarks: [] },
    pose: { landmarks: [] }
  });
  // State just for the dev UI (updated periodically or on demand)
  // const [uiPerceptionState, setUiPerceptionState] = useState<PerceptionData>(perceptionDataRef.current);
  
  const [isProcessing, setIsProcessing] = useState(false);
  const isProcessingRef = useRef<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const lastGestureResultRef = useRef<GestureResult | null>(null);
  const lastInterpretedGestureRef = useRef<string | null>(null);
  
  const recognizerRef = useRef(new GestureRecognizer());
  const requestRef = useRef<number>(0);
  const cameraActiveRef = useRef<boolean>(false);
  const observationQueueRef = useRef<ActionObservationPayload[]>([]);
  const lastVideoTimeRef = useRef<number>(-1);
  const frameCounterRef = useRef<number>(0);

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    const initializeMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });
        setHandLandmarker(landmarker);
        
        const faceLM = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
          numFaces: 1
        });
        setFaceLandmarker(faceLM);

        const poseLM = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 1
        });
        setPoseLandmarker(poseLM);

      } catch (err) {
        console.error("Failed to initialize MediaPipe:", err);
        setError("Failed to load AI models.");
      }
    };
    initializeMediaPipe();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setCameraActive(true);
          cameraActiveRef.current = true;
          setError(null);
          
          // UI updater for Dev Panel was removed

          // Start detection loop once video is loaded
          predictWebcam();
        };
      }
    } catch (err) {
      setError("Could not access camera. Please allow permissions.");
      console.error(err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
      cameraActiveRef.current = false;
      
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }
      
      // Clear canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      }
    }
  };

  const predictWebcam = () => {
    if (!videoRef.current || !canvasRef.current || !handLandmarker) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const canvasCtx = canvas.getContext("2d");
    
    if (!canvasCtx) return;

    // Ensure canvas and video dimensions match intrinsic video size
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      video.width = video.videoWidth;
      video.height = video.videoHeight;
    }

    let startTimeMs = performance.now();
    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      frameCounterRef.current++;
      const currentFrame = frameCounterRef.current;
      
      const results = handLandmarker.detectForVideo(video, startTimeMs);
      
      // Update Hands ref
      perceptionDataRef.current.hands = {
        landmarks: results.landmarks || [],
        handednesses: results.handednesses ? results.handednesses.map(h => h[0].categoryName) : []
      };

      // Interleaved scheduling: Face on even frames, Pose on odd frames
      if (currentFrame % 2 === 0 && faceLandmarker) {
          const faceResults = faceLandmarker.detectForVideo(video, startTimeMs);
          if (faceResults.faceLandmarks) {
              perceptionDataRef.current.face.landmarks = faceResults.faceLandmarks;
          }
      } else if (currentFrame % 2 === 1 && poseLandmarker) {
          const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
          console.log("PoseResults:", poseResults);
          if (poseResults.landmarks && poseResults.landmarks.length > 0) {
              perceptionDataRef.current.pose.landmarks = poseResults.landmarks;
          } else {
              perceptionDataRef.current.pose.landmarks = [];
          }
      }
      
      // Multi-hand processing
      const numHands = results.landmarks ? results.landmarks.length : 0;
      const handednesses = results.handednesses ? results.handednesses.map(h => h[0].categoryName) : [];
      let distanceNormalized = null;
      
      // Action Observer process
      const observation = actionObserverRef.current.processFrames(perceptionDataRef.current, startTimeMs);
      if (observation) {
         handleObservation(observation);
      }
      
      if (numHands === 2) {
        // Calculate distance between wrists (landmark index 0)
        const wrist1 = results.landmarks[0][0];
        const wrist2 = results.landmarks[1][0];
        const dx = wrist1.x - wrist2.x;
        const dy = wrist1.y - wrist2.y;
        const dz = wrist1.z - wrist2.z;
        distanceNormalized = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        // Smoothly interpolate the scale
        // Sensible limits: distance between 0.1 (hands close) and 0.8 (hands wide apart)
        // Map to scale between 0.5 and 3.5
        let targetScale = ((distanceNormalized - 0.1) / 0.7) * (3.5 - 0.5) + 0.5;
        targetScale = Math.max(0.5, Math.min(targetScale, 3.5));
        
        // Apply smoothing (lerp)
        smoothedDistanceRef.current += (targetScale - smoothedDistanceRef.current) * 0.1;
        
        // --- Phase 3: Position Control ---
        const midX = (wrist1.x + wrist2.x) / 2;
        const midY = (wrist1.y + wrist2.y) / 2;
        
        // Convert to percentages. Since video is mirrored, we might need to invert X if it feels backward.
        // MediaPipe returns X from 0 (left of image) to 1 (right of image).
        // Since we mirrored the image via CSS, we should invert X for correct physical mapping:
        const targetX = (1.0 - midX) * 100;
        const targetY = midY * 100;
        
        // Lerp position
        smoothedPosXRef.current += (targetX - smoothedPosXRef.current) * 0.1;
        smoothedPosYRef.current += (targetY - smoothedPosYRef.current) * 0.1;
        
        if (orbRef.current) {
          orbRef.current.style.opacity = '1';
          orbRef.current.style.left = `${smoothedPosXRef.current}%`;
          orbRef.current.style.top = `${smoothedPosYRef.current}%`;
          orbRef.current.style.transform = `translate(-50%, -50%) scale(${smoothedDistanceRef.current})`;
        }
      } else {
        if (orbRef.current) {
          orbRef.current.style.opacity = '0';
        }
      }
      
      setHandTrackingData({
        numHands,
        handednesses,
        distanceNormalized
      });

      const result = recognizerRef.current.processFrames(results.landmarks, startTimeMs);
      
      const lastRes = lastGestureResultRef.current;
      if (!lastRes || 
          lastRes.gesture !== result.gesture || 
          Math.abs(lastRes.confidence - result.confidence) > 0.05 || 
          lastRes.stable !== result.stable) {
        setGestureResult(result);
        lastGestureResultRef.current = result;
      }

      // If gesture becomes stable, and it's a new gesture, trigger agent
      if (result.stable && result.gesture !== 'NO_HAND' && result.gesture !== 'UNKNOWN') {
        if (lastInterpretedGestureRef.current !== result.gesture && !isProcessingRef.current) {
          lastInterpretedGestureRef.current = result.gesture;
          actionObserverRef.current.addSemanticObservation(result.gesture, startTimeMs);
          
          const def = GESTURE_VOCABULARY[result.gesture];
          if (def && def.directSpeech) {
            // Fast-lane for deterministic actions (e.g. HELP, numbers)
            interpretGesture(result);
          }
        }
      } else if (!result.stable || result.gesture === 'NO_HAND') {
        // Reset so same gesture can be interpreted again if hand is removed and brought back
        if (result.gesture === 'NO_HAND') {
           lastInterpretedGestureRef.current = null;
        }
      }

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      
      const drawingUtils = new DrawingUtils(canvasCtx);
      const pd = perceptionDataRef.current;
      
      // Draw Pose (Background layer)
      if (pd.pose.landmarks && pd.pose.landmarks.length > 0) {
        for (const landmark of pd.pose.landmarks) {
          drawingUtils.drawLandmarks(landmark, { color: 'rgba(255, 255, 255, 0.4)', radius: 3 });
          drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: 'rgba(255, 255, 255, 0.3)', lineWidth: 2 });
        }
      }
      
      // Draw Face (Mid layer)
      if (pd.face.landmarks && pd.face.landmarks.length > 0) {
        for (const landmark of pd.face.landmarks) {
          drawingUtils.drawConnectors(landmark, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: 'rgba(192, 192, 192, 0.2)', lineWidth: 1 });
          drawingUtils.drawConnectors(landmark, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: 'rgba(255, 138, 101, 0.5)' });
          drawingUtils.drawConnectors(landmark, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: 'rgba(255, 138, 101, 0.5)' });
          drawingUtils.drawConnectors(landmark, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: 'rgba(224, 224, 224, 0.5)' });
          drawingUtils.drawConnectors(landmark, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: 'rgba(255, 138, 101, 0.5)' });
        }
      }

      // Draw Hands (Foreground layer)
      if (pd.hands.landmarks && pd.hands.landmarks.length > 0) {
        for (const landmarks of pd.hands.landmarks) {
          drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
            color: "#00FF00",
            lineWidth: 5
          });
          drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
        }
      }
      
      canvasCtx.restore();
    }
    
    // Continue loop
    if (cameraActiveRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
    }
  };

  const processObservationQueue = async () => {
    if (isProcessingRef.current || observationQueueRef.current.length === 0) return;
    
    isProcessingRef.current = true;
    setIsProcessing(true);
    
    while (observationQueueRef.current.length > 0) {
      const payload = observationQueueRef.current.shift()!;
      
      try {
        setError(null);
        
        const reqTime = Date.now();
        
        const response = await fetch(`${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'}/api/agent/observe`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
          throw new Error("Backend API error");
        }
        
        const data: AgentDecision = await response.json();
        const resTime = Date.now();
        
        console.log(`[OBSERVATION] ID: ${payload.observationId}`);
        console.log(`- Observations: ${payload.observations.join(", ")}`);
        console.log(`- Request Time: ${new Date(reqTime).toISOString()}`);
        console.log(`- Response Time: ${new Date(resTime).toISOString()} (${resTime - reqTime}ms)`);
        console.log(`- Response: Intent=${data.intent}, Decision=${data.decision}, Text="${data.response_text}"`);
        
        // Atomic UI Update: only update both if they are successfully returned together
        setActionPayload(payload);
        setAgentDecision(data);
        
        if ((data.decision === "SPEAK" || data.decision === "CONFIRM") && data.response_text) {
          playTTS(data.response_text);
        }
      } catch (err) {
        console.error(err);
        setError("Failed to interpret action observation.");
      }
    }
    
    setIsProcessing(false);
    isProcessingRef.current = false;
  };

  const handleObservation = (payload: ActionObservationPayload) => {
    observationQueueRef.current.push(payload);
    processObservationQueue();
  };

  // Restart loop if camera is active but loop stopped (e.g., model loaded after camera started)
  useEffect(() => {
    if (cameraActiveRef.current && handLandmarker && videoRef.current && videoRef.current.readyState >= 2) {
      predictWebcam();
    }
  }, [handLandmarker]);

  const playTTS = (text: string) => {
    if (!window.speechSynthesis) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);
    
    window.speechSynthesis.speak(utterance);
  };

  const interpretGesture = async (result: GestureResult) => {
    isProcessingRef.current = true;
    setIsProcessing(true);
    try {
      const apiUrl = `${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001'}/api/agent/interpret`;
      
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          gesture: result.gesture,
          confidence: result.confidence,
          stable: result.stable,
          timestamp: Date.now()
        })
      });
      
      if (!response.ok) throw new Error('API Error');
      
      const data: AgentDecision = await response.json();
      setAgentDecision(data);
      
      if ((data.decision === 'SPEAK' || data.decision === 'CONFIRM') && data.response_text) {
        playTTS(data.response_text);
      }
    } catch (err) {
      console.error("Agent interpretation failed:", err);
    } finally {
      isProcessingRef.current = false;
      setIsProcessing(false);
    }
  };

  useEffect(() => {
    return () => stopCamera(); // Cleanup on unmount
  }, []);

  return (
    <div className="app-container">
      <img src="/robot-mascot.jpg" alt="Futuristic Robot Mascot" className="robot-mascot" />

      {/* Left Sidebar */}
      <aside className="sidebar-left">
        <div className="logo">
          <span className="logo-icon">|||</span>
          <div className="logo-text">
            <h1>SilentSpeak</h1>
            <p className="subtitle">Speak Through Silence</p>
          </div>
        </div>

        <nav className="nav-menu">
          <div className="nav-item active"><span className="icon">🏠</span> Home</div>
          <div className="nav-item"><span className="icon">✌️</span> Gestures</div>
          <div className="nav-item"><span className="icon">💬</span> Chat</div>
          <div className="nav-item"><span className="icon">🕒</span> History</div>
          <div className="nav-item"><span className="icon">⚙️</span> Settings</div>
          <div className="nav-item"><span className="icon">❓</span> Help</div>
        </nav>

        <div className="tip-panel">
          <div className="tip-header"><span className="icon">💡</span> Tip</div>
          <p>Show clear hand gestures in good lighting for best results.</p>
        </div>
      </aside>

      {/* Main Center Content */}
      <main className="main-content">
        <div className="glass-panel video-container">
          <div className="video-header">
            <div className="video-title">
              <h2>Live Camera</h2>
              <span className="live-dot"></span>
            </div>
            <div className="camera-selector">Camera: Integrated Webcam <span>▼</span></div>
          </div>
          
          {error && <div className="error-message">{error}</div>}
          
          <div className={`video-wrapper ${cameraActive ? 'active' : ''}`}>
            {!cameraActive && (
              <div className="placeholder-content">
                <span className="camera-icon">📷</span>
                <p className="camera-status-text">Camera is inactive</p>
                <p className="camera-sub-text">Show your gesture...</p>
              </div>
            )}
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className={`camera-feed ${cameraActive ? 'visible' : 'hidden'}`}
            />
            <canvas
              ref={canvasRef}
              className={`output-canvas ${cameraActive ? 'visible' : 'hidden'}`}
            />
          </div>

          <div className="controls">
            {!cameraActive ? (
              <button className="btn btn-primary" onClick={startCamera} disabled={!handLandmarker}>
                <span className="btn-icon">▶</span> {handLandmarker ? 'Start Camera' : 'Loading ML Model...'}
              </button>
            ) : (
              <button className="btn btn-danger" onClick={stopCamera}>
                <span className="btn-icon">⏹</span> Stop Camera
              </button>
            )}
            
            {/* The Futuristic Energy Orb */}
            <div ref={orbRef} className="energy-orb">
              <div className="orb-core"></div>
              <div className="orb-ring ring-1"></div>
              <div className="orb-ring ring-2"></div>
              <div className="orb-particles"></div>
            </div>
          </div>
        </div>

        <div className="glass-panel recognition-panel horizontal">
          <div className="recognition-info">
            <span className="gesture-label">Recognized Gesture</span>
            <div className="gesture-value">
              {gestureResult?.gesture === 'NO_HAND' ? 'NO HAND DETECTED' : 
               gestureResult?.gesture === 'UNKNOWN' ? 'UNKNOWN GESTURE' : 
               gestureResult?.gesture || 'WAITING...'}
            </div>
          </div>
          <div className={`status-badge ${gestureResult?.stable ? 'detected' : 'pending'}`}>
            {gestureResult?.stable ? '✓ Detected' : '...'}
          </div>
        </div>

        <div className="glass-panel speech-panel horizontal">
          <div className="speech-info">
            <span className="gesture-label">Speech Output</span>
            <div className="speech-text">
               <span className="speaker-icon">🔊</span>
               <span>{agentDecision ? agentDecision.response_text : "Waiting for stable gesture..."}</span>
            </div>
          </div>
          <div className="speech-actions">
            <button className="icon-btn" title="Copy Text">📋</button>
            <button className="btn btn-primary" onClick={() => {if(agentDecision?.response_text) playTTS(agentDecision.response_text)}}>
              <span className="speaker-icon">🔊</span> Speak
            </button>
          </div>
        </div>
      </main>

      {/* Right Sidebar */}
      <aside className="sidebar-right">
        <div className="top-status">
          <div className="status-badge online"><span className="dot"></span> Online</div>
        </div>
        
        <div className="glass-panel action-menu">
           <h2>Quick Actions</h2>
           <div className="action-item"><span className="icon">🗑️</span> Clear Text</div>
           <div className="action-item"><span className="icon">📋</span> Copy Text</div>
           <div className="action-item"><span className="icon">📥</span> Download</div>
        </div>
        
        <div className="glass-panel recent-gestures">
           <h2>Agent Status & Context</h2>
           <div className="agent-status">
              {isProcessing ? (
                <span className="agent-badge processing">Thinking...</span>
              ) : isSpeaking ? (
                <span className="agent-badge speaking">🔊 Speaking...</span>
              ) : (
                <span className="agent-badge idle">Idle</span>
              )}
           </div>

           {actionPayload && actionPayload.observations.length > 0 && (
               <div className="observation-box">
                 <h4>Recent Observations</h4>
                 <ul>
                    {actionPayload.observations.map((obs, idx) => (
                       <li key={idx}>{obs}</li>
                    ))}
                 </ul>
               </div>
            )}
           
           <button className="view-history-btn">View all history →</button>
        </div>

        <div className="glass-panel dev-panel" style={{display: 'none'}}>
           {/* Keeping Dev info hidden but preserved for functionality */}
           <h2>System Dev Status</h2>
           <div className="status-item">
             <span>Camera: {cameraActive ? 'Active' : 'Inactive'}</span>
             <span>Hands: {handTrackingData.numHands}</span>
           </div>
        </div>
      </aside>
    </div>
  )
}

export default App
