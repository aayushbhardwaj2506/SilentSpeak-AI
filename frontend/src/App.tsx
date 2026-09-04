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

interface HistoryItem {
  id: string;
  gesture: string;
  text: string;
  timestamp: number;
}

interface HandTrackingData {
  numHands: number;
  hands: { category: string; score: number }[];
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
  const [_, setError] = useState<string | null>(null);
  const [requiresTap, setRequiresTap] = useState(false);
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const [faceLandmarker, setFaceLandmarker] = useState<FaceLandmarker | null>(null);
  const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | null>(null);
  
  const [leftGestureResult, setLeftGestureResult] = useState<GestureResult | null>(null);
  const [rightGestureResult, setRightGestureResult] = useState<GestureResult | null>(null);
  const [agentDecision, setAgentDecision] = useState<AgentDecision | null>(null);
  const [combinedHistory, setCombinedHistory] = useState<HistoryItem[]>([]);
  const [handTrackingData, setHandTrackingData] = useState<HandTrackingData>({ numHands: 0, hands: [], distanceNormalized: null });
  
  const actionObserverRef = useRef(new ActionObserver());
  const [___, setActionPayload] = useState<ActionObservationPayload | null>(null);
  const [liveObservations, setLiveObservations] = useState<{left: string[], right: string[], conclusion: string[]}>({left: [], right: [], conclusion: []});

  // Use a ref for real-time perception data to avoid 60fps re-renders
  const perceptionDataRef = useRef<PerceptionData>({
    hands: { landmarks: [], handednesses: [] },
    face: { landmarks: [] },
    pose: { landmarks: [] }
  });
  
  const [isProcessing, setIsProcessing] = useState(false);
  const isProcessingRef = useRef<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  
  const lastLeftResultRef = useRef<GestureResult | null>(null);
  const lastRightResultRef = useRef<GestureResult | null>(null);
  const lastInterpretedLeftRef = useRef<string | null>(null);
  const lastInterpretedRightRef = useRef<string | null>(null);
  
  const leftRecognizerRef = useRef(new GestureRecognizer());
  const rightRecognizerRef = useRef(new GestureRecognizer());
  
  const requestRef = useRef<number>(0);
  const cameraActiveRef = useRef<boolean>(false);
  const observationQueueRef = useRef<ActionObservationPayload[]>([]);
  const lastVideoTimeRef = useRef<number>(-1);
  const frameCounterRef = useRef<number>(0);

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    actionObserverRef.current.onObservationsUpdate = (obs) => {
      setLiveObservations(obs);
    };

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
      console.log("[CAMERA] getUserMedia requested");
      let stream: MediaStream;
      try {
        // Try front camera first
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { facingMode: { ideal: "user" } }, 
          audio: false 
        });
      } catch (e) {
        // Fallback to any camera
        console.log("[CAMERA] Fallback to any camera");
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }

      console.log("[CAMERA] stream obtained:", stream?.id);
      const videoTrack = stream.getVideoTracks()[0];
      console.log("[CAMERA] camera track status:", videoTrack ? videoTrack.readyState : 'no track', videoTrack?.label);

      if (videoRef.current) {
        const video = videoRef.current;
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;

        const handleVideoLoaded = async () => {
          console.log("[CAMERA] onloadedmetadata fired. Video dimensions:", video.videoWidth, "x", video.videoHeight);
          console.log("[CAMERA] video.readyState:", video.readyState);
          
          try {
            await video.play();
            console.log("[CAMERA] video.play() success");
            setCameraActive(true);
            cameraActiveRef.current = true;
            setError(null);
            setRequiresTap(false);
            
            // Start detection loop once video is loaded
            predictWebcam();
          } catch (e) {
            console.error("[CAMERA] video.play() failed:", e);
            setRequiresTap(true);
            setCameraActive(true);
            cameraActiveRef.current = true;
            setError("Tap to Start Camera (Autoplay restricted)");
          }
        };

        video.onloadedmetadata = handleVideoLoaded;
        video.srcObject = stream;

        if (video.readyState >= 1) {
          handleVideoLoaded();
        }
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
      let distanceNormalized = null;
      
      // Update Hand Tracking Stats for UI
      setHandTrackingData({
        numHands: numHands,
        hands: results.handednesses ? results.handednesses.map(h => ({
          category: h[0].categoryName,
          score: h[0].score
        })) : [],
        distanceNormalized: distanceNormalized
      });
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

      const leftLandmarks: any[][] = [];
      const rightLandmarks: any[][] = [];
      
      if (results.landmarks && results.handednesses) {
        for (let i = 0; i < results.landmarks.length; i++) {
          const cat = results.handednesses[i][0].categoryName.toLowerCase();
          if (cat === 'left') leftLandmarks.push(results.landmarks[i]);
          else rightLandmarks.push(results.landmarks[i]);
        }
      }
      
      const leftResult = leftRecognizerRef.current.processFrames(leftLandmarks, startTimeMs);
      const rightResult = rightRecognizerRef.current.processFrames(rightLandmarks, startTimeMs);

      const checkStateChange = (res: GestureResult, lastRef: React.MutableRefObject<GestureResult | null>) => {
         return !lastRef.current || 
                lastRef.current.gesture !== res.gesture || 
                Math.abs(lastRef.current.confidence - res.confidence) > 0.05 || 
                lastRef.current.stable !== res.stable;
      };

      if (checkStateChange(leftResult, lastLeftResultRef)) {
        setLeftGestureResult(leftResult);
        lastLeftResultRef.current = leftResult;
      }
      if (checkStateChange(rightResult, lastRightResultRef)) {
        setRightGestureResult(rightResult);
        lastRightResultRef.current = rightResult;
      }

      const checkStable = (res: GestureResult, lastRef: React.MutableRefObject<string | null>, category: 'left'|'right') => {
        if (res.stable && res.gesture !== 'NO_HAND' && res.gesture !== 'UNKNOWN') {
          if (lastRef.current !== res.gesture && !isProcessingRef.current) {
            lastRef.current = res.gesture;
            actionObserverRef.current.addSemanticObservation(category, res.gesture, startTimeMs);
            return true;
          }
        } else if (!res.stable || res.gesture === 'NO_HAND') {
          if (res.gesture === 'NO_HAND') lastRef.current = null;
        }
        return false;
      };

      const leftTriggered = checkStable(leftResult, lastInterpretedLeftRef, 'left');
      const rightTriggered = checkStable(rightResult, lastInterpretedRightRef, 'right');
      
      if (leftTriggered || rightTriggered) {
         const lG = lastInterpretedLeftRef.current;
         const rG = lastInterpretedRightRef.current;
         let combinedGesture = "";
         if (lG && rG) combinedGesture = `LEFT: ${lG}, RIGHT: ${rG}`;
         else if (lG) combinedGesture = `LEFT: ${lG}`;
         else if (rG) combinedGesture = `RIGHT: ${rG}`;
         
         if (combinedGesture && !isProcessingRef.current) {
            const defL = lG ? GESTURE_VOCABULARY[lG] : null;
            const defR = rG ? GESTURE_VOCABULARY[rG] : null;
            if ((defL && defL.directSpeech) || (defR && defR.directSpeech)) {
               interpretGesture({ gesture: combinedGesture as any, confidence: Math.max(leftResult.confidence, rightResult.confidence), stable: true, timestamp: startTimeMs });
            }
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
          setCombinedHistory(prev => [{
            id: Date.now().toString() + Math.random(),
            gesture: 'SEQUENCE',
            text: data.response_text,
            timestamp: Date.now()
          }, ...prev].slice(0, 5));
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
        setCombinedHistory(prev => [{
          id: Date.now().toString() + Math.random(),
          gesture: result.gesture,
          text: data.response_text,
          timestamp: Date.now()
        }, ...prev].slice(0, 5));
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
      <div className="baymax-bg"></div>
      
      {/* Cinematic Header */}
      <header className="cinematic-header">
        <div className="brand">
          <h1>SilentSpeak <span>AI</span></h1>
          <p>Gesture Interpretation Engine</p>
        </div>
      </header>

      {/* Main Layout */}
      <main className="main-layout">
        
        {/* Hero Camera Section */}
        <section className="camera-section">
          {cameraActive && (
            <div className="camera-overlay-ui">
              <div className="live-indicator"><span className="dot"></span> LIVE</div>
              {handTrackingData.numHands > 0 && (
                <div className="live-indicator" style={{background: 'rgba(220,38,38,0.2)', borderColor: 'var(--accent)'}}>
                  HAND TRACKING ACTIVE
                </div>
              )}
            </div>
          )}
          
          <div className="video-wrapper">
            {!cameraActive && (
              <div style={{color: 'var(--text-muted)', letterSpacing: '2px', textAlign: 'center', zIndex: 5}}>
                <div style={{fontSize: '3rem', marginBottom: '1rem'}}>📷</div>
                <p>CAMERA OFFLINE</p>
              </div>
            )}
            
            {requiresTap && (
              <div 
                style={{
                  position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                  display: 'flex', justifyContent: 'center', alignItems: 'center',
                  backgroundColor: 'rgba(0,0,0,0.8)', zIndex: 20, cursor: 'pointer', flexDirection: 'column', gap: '1rem'
                }}
                onClick={() => {
                  videoRef.current?.play().then(() => {
                    setRequiresTap(false); setError(null); predictWebcam();
                  });
                }}
              >
                <div style={{ padding: '1rem 2rem', background: 'var(--accent)', color: '#000', borderRadius: '4px', fontWeight: 'bold', letterSpacing: '2px' }}>
                  TAP TO START CAMERA
                </div>
              </div>
            )}
            
            <video ref={videoRef} autoPlay playsInline muted className={`camera-feed ${cameraActive ? 'visible' : 'hidden'}`} />
            <canvas ref={canvasRef} className={`output-canvas ${cameraActive ? 'visible' : 'hidden'}`} />
            <div ref={orbRef} className="energy-orb"><div className="orb-core"></div></div>
          </div>

          {/* Two-Hand Visualization Footer */}
          {cameraActive && (
            <div className="hand-tracking-overlay">
              <div className="hands-connection-display">
                <div className={`hand-label ${handTrackingData.hands.some(h => h.category === 'Left') ? 'active' : ''}`}>LEFT HAND</div>
                <div className={`hand-connector ${handTrackingData.numHands === 2 ? 'active' : ''}`}></div>
                <div className={`hand-label ${handTrackingData.hands.some(h => h.category === 'Right') ? 'active' : ''}`}>RIGHT HAND</div>
              </div>
            </div>
          )}
        </section>

        {/* AI Interpreter Panel */}
        <aside className="interpreter-panel" style={{ width: '420px', overflowY: 'auto' }}>
          
          {/* Avatar Section */}
          <div className="avatar-section">
            <div className={`avatar-character ${isSpeaking ? 'speaking' : 'idle'}`}>
              <div className="avatar-eyes">
                <div className="eye"></div>
                <div className="eye"></div>
              </div>
              <div className="avatar-mouth"></div>
            </div>
            <div className={`ai-status-text ${isProcessing || isSpeaking ? 'active' : ''}`}>
              {isProcessing ? "Analyzing Gesture Sequence..." : isSpeaking ? "Speaking..." : "AI Interpreter Idle"}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', width: '100%', marginBottom: '1rem' }}>
            {/* Left Arm Panel */}
            <div style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '0.5rem' }}>
              <div className="section-label" style={{textAlign: 'center', marginBottom: '0.5rem'}}>Left Arm</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--accent)', marginBottom: '0.5rem', textAlign: 'center', minHeight: '15px' }}>
                Flow: {leftGestureResult?.gesture && leftGestureResult.gesture !== 'NO_HAND' && leftGestureResult.gesture !== 'UNKNOWN' ? (GESTURE_VOCABULARY[leftGestureResult.gesture]?.displayName || leftGestureResult.gesture) : '...'}
              </div>
              <div className="live-observations" style={{ height: '120px', overflowY: 'auto' }}>
                {liveObservations.left.length > 0 ? liveObservations.left.map((obs, idx) => (
                  <div key={idx} style={{ fontSize: '0.65rem', color: '#e2e8f0', borderLeft: '2px solid var(--accent)', paddingLeft: '4px', marginBottom: '2px', lineHeight: '1.2' }}>&gt; {obs}</div>
                )) : <div style={{color: 'var(--text-muted)', fontSize: '0.65rem', fontStyle: 'italic', textAlign: 'center', marginTop: '1rem'}}>Waiting...</div>}
              </div>
            </div>
            
            {/* Right Arm Panel */}
            <div style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '8px', padding: '0.5rem' }}>
              <div className="section-label" style={{textAlign: 'center', marginBottom: '0.5rem'}}>Right Arm</div>
              <div style={{ fontSize: '0.65rem', color: 'var(--accent)', marginBottom: '0.5rem', textAlign: 'center', minHeight: '15px' }}>
                Flow: {rightGestureResult?.gesture && rightGestureResult.gesture !== 'NO_HAND' && rightGestureResult.gesture !== 'UNKNOWN' ? (GESTURE_VOCABULARY[rightGestureResult.gesture]?.displayName || rightGestureResult.gesture) : '...'}
              </div>
              <div className="live-observations" style={{ height: '120px', overflowY: 'auto' }}>
                {liveObservations.right.length > 0 ? liveObservations.right.map((obs, idx) => (
                  <div key={idx} style={{ fontSize: '0.65rem', color: '#e2e8f0', borderLeft: '2px solid var(--accent)', paddingLeft: '4px', marginBottom: '2px', lineHeight: '1.2' }}>&gt; {obs}</div>
                )) : <div style={{color: 'var(--text-muted)', fontSize: '0.65rem', fontStyle: 'italic', textAlign: 'center', marginTop: '1rem'}}>Waiting...</div>}
              </div>
            </div>
          </div>

          {/* AI Conclusion & Understanding */}
          <div className="interpretation-section" style={{marginBottom: '1rem'}}>
            <div className="section-label">Combined AI Conclusion</div>
            <div className="live-observations" style={{ maxHeight: '100px', overflowY: 'auto', marginBottom: '0.75rem' }}>
               {liveObservations.conclusion.length > 0 ? liveObservations.conclusion.map((obs, idx) => (
                  <div key={idx} style={{ fontSize: '0.7rem', color: '#10b981', borderLeft: '2px solid #10b981', paddingLeft: '6px', marginBottom: '4px', lineHeight: '1.3' }}>&gt; {obs}</div>
               )) : <div style={{color: 'var(--text-muted)', fontSize: '0.7rem', fontStyle: 'italic'}}>No sequence formed yet...</div>}
            </div>

            {agentDecision && agentDecision.intent ? (
              <>
                <div className="semantic-flow" style={{ fontSize: '0.75rem' }}>
                  <span className="flow-gesture">{agentDecision.intent}</span>
                  <span className="flow-arrow">→</span>
                  <span>UNDERSTANDING</span>
                </div>
                <div className="final-speech" style={{ fontSize: '0.9rem', marginTop: '0.5rem' }}>
                  "{agentDecision.response_text}"
                </div>
                <div className="confidence-display" style={{ marginTop: '0.5rem' }}>
                  <span>Confidence {Math.round((agentDecision.confidence || 0.91) * 100)}%</span>
                  <div className="conf-bar-bg">
                    <div className="conf-bar-fill" style={{width: `${Math.round((agentDecision.confidence || 0.91) * 100)}%`}}></div>
                  </div>
                </div>
              </>
            ) : (
              <div style={{color: 'var(--text-muted)', fontSize: '0.75rem', fontStyle: 'italic', textAlign: 'center'}}>Waiting for combined gestures...</div>
            )}
          </div>

          {/* History */}
          <div className="history-section">
             <div className="section-label">Recent Conversation</div>
             <div className="history-list">
               {combinedHistory.length === 0 ? (
                 <div style={{color: 'var(--text-muted)', fontSize: '0.8rem', fontStyle: 'italic'}}>No history yet</div>
               ) : combinedHistory.map(item => (
                 <div key={item.id} className="history-bubble">
                   <div style={{fontSize: '0.65rem', color: 'var(--accent)', marginBottom: '0.3rem', letterSpacing: '1px'}}>
                     {item.gesture === 'SEQUENCE' ? 'SEQUENCE' : item.gesture}
                   </div>
                   {item.text}
                 </div>
               ))}
             </div>
          </div>
        </aside>
      </main>

      {/* Unified Dock Controls */}
      <footer className="controls-dock">
        <div className="dock-status">
          <div className={`status-item ${cameraActive ? 'active' : ''}`}>
             <div className="indicator"></div> CAMERA
          </div>
          <div className={`status-item ${cameraActive ? 'active' : ''}`}>
             <div className="indicator"></div> AI ENGINE
          </div>
          <div className={`status-item ${handTrackingData.numHands > 0 ? 'active' : ''}`}>
             <div className="indicator"></div> TRACKING
          </div>
        </div>
        
        {!cameraActive ? (
          <button className="btn-dock primary" onClick={startCamera} disabled={!handLandmarker}>
            START INTERPRETER
          </button>
        ) : (
          <button className="btn-dock danger" onClick={stopCamera}>
            ■ STOP SYSTEM
          </button>
        )}
      </footer>
    </div>
  )
}

export default App
