import { GESTURE_VOCABULARY } from './GestureVocabulary';

export interface ActionObservationPayload {
  observationId: string;
  duration: number;
  observations: string[];
}

export interface PerceptionData {
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

export class ActionObserver {
  private isObserving: boolean = false;
  private observationStartTime: number = 0;
  private lastMovementTime: number = 0;
  
  private observations: string[] = []; // Changed to array for sequence order
  
  // To track movement speed/distance
  private lastHandPosition: { x: number, y: number, z: number } | null = null;
  private accumulatedMovement: number = 0;

  // Thresholds
  private readonly MOVEMENT_THRESHOLD = 0.05;
  private readonly PAUSE_THRESHOLD_MS = 1000;
  private readonly MAX_OBSERVATION_TIME_MS = 8000;
  
  private addObservationSafe(obs: string) {
    if (this.observations.length === 0 || this.observations[this.observations.length - 1] !== obs) {
      this.observations.push(obs);
    }
  }

  public processFrames(perceptionData: PerceptionData, timestampMs: number): ActionObservationPayload | null {
    const hasHands = perceptionData.hands.landmarks.length > 0;
    
    // Feature Extraction
    if (hasHands) {
      const primaryHand = perceptionData.hands.landmarks[0];
      const wrist = primaryHand[0];
      const indexTip = primaryHand[8];
      const thumbTip = primaryHand[4];
      
      // Calculate movement
      if (this.lastHandPosition) {
        const dx = wrist.x - this.lastHandPosition.x;
        const dy = wrist.y - this.lastHandPosition.y;
        const dz = wrist.z - this.lastHandPosition.z;
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
        
        if (dist > 0.01) {
          this.lastMovementTime = timestampMs;
          this.accumulatedMovement += dist;
        }
        
        // Start observing if movement exceeds threshold
        if (!this.isObserving && this.accumulatedMovement > this.MOVEMENT_THRESHOLD) {
          this.isObserving = true;
          this.observationStartTime = timestampMs;
          this.observations = [];
        }
      }
      this.lastHandPosition = { x: wrist.x, y: wrist.y, z: wrist.z };
      
      // Extract Semantics if observing
      if (this.isObserving) {
        // Feature 1: Pointing (Fallbacks)
        const indexExtendedDist = Math.sqrt(
           (indexTip.x - wrist.x)**2 + (indexTip.y - wrist.y)**2
        );
        const thumbDist = Math.sqrt(
           (thumbTip.x - wrist.x)**2 + (thumbTip.y - wrist.y)**2
        );
        
        if (indexExtendedDist > 0.25 && thumbDist < 0.2) {
           if (indexTip.x < wrist.x - 0.1) this.addObservationSafe("user pointed to the right");
           else if (indexTip.x > wrist.x + 0.1) this.addObservationSafe("user pointed to the left");
           else if (indexTip.y < wrist.y - 0.1) this.addObservationSafe("user pointed upward");
           else if (indexTip.y > wrist.y + 0.1) this.addObservationSafe("user pointed downward");
           else this.addObservationSafe("user pointed forward");
        }
        
        // Feature 2: Hand to Mouth
        if (perceptionData.face.landmarks && perceptionData.face.landmarks.length > 0) {
           const mouthTop = perceptionData.face.landmarks[0][13];
           const mouthBottom = perceptionData.face.landmarks[0][14];
           const mouthY = (mouthTop.y + mouthBottom.y) / 2;
           const mouthX = (mouthTop.x + mouthBottom.x) / 2;
           
           const handToMouthDist = Math.sqrt((wrist.x - mouthX)**2 + (wrist.y - mouthY)**2);
           if (handToMouthDist < 0.25) { 
               this.addObservationSafe("right hand moved toward mouth");
           }
        }
        
        // Feature 3: Waving / Significant motion
        if (this.accumulatedMovement > 1.5) { 
           this.addObservationSafe("user made a significant waving motion");
        }
      }
    }
    
    // Flush Logic (Natural Pause)
    if (this.isObserving) {
       const timeSinceLastMovement = timestampMs - this.lastMovementTime;
       const timeObserving = timestampMs - this.observationStartTime;
       
       if (timeSinceLastMovement > this.PAUSE_THRESHOLD_MS || timeObserving > this.MAX_OBSERVATION_TIME_MS) {
          const payload = this.flushObservation(timestampMs);
          if (payload && payload.observations.length > 0) {
             return payload;
          }
       }
    }
    
    return null;
  }
  
  private flushObservation(timestampMs: number): ActionObservationPayload | null {
    if (!this.isObserving) return null;
    
    const duration = (timestampMs - this.observationStartTime) / 1000.0;
    const finalObservations = [...this.observations];
    
    // Reset
    this.isObserving = false;
    this.observations = [];
    this.accumulatedMovement = 0;
    
    if (finalObservations.length === 0) return null;
    
    const observationId = `obs_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    
    return {
      observationId,
      duration,
      observations: finalObservations
    };
  }

  public addSemanticObservation(token: string, timestampMs: number) {
    if (!this.isObserving) {
      this.isObserving = true;
      this.observationStartTime = timestampMs;
      this.observations = [];
      this.accumulatedMovement = 0;
    }
    
    const def = GESTURE_VOCABULARY[token];
    if (def) {
      const sentence = `[GESTURE: ${def.displayName}] Potential meanings: ${def.possibleMeanings.join(", ")}`;
      this.addObservationSafe(sentence);
      this.lastMovementTime = timestampMs;
    }
  }
}
