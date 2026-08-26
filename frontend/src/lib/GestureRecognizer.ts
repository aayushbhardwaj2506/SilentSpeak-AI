export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}
export type GestureType = 'HELLO' | 'YES' | 'NO' | 'HELP' | 'WATER' | 'STOP' | 'THANK_YOU' | 'UNKNOWN' | 'NO_HAND' | 
  'THUMBS_UP' | 'THUMBS_DOWN' | 'PEACE' | 'OK_SIGN' | 'FIST' | 'OPEN_PALM' |
  'POINT_UP' | 'POINT_DOWN' | 'POINT_LEFT' | 'POINT_RIGHT' | 
  'PALMS_TOGETHER' | 'PINCH' | 'CALL_ME' | 'INDEX_HOOK' | 'CUPPED_HAND' | 'FLAT_HAND_UP' | 'HAND_ON_CHEST';

export interface GestureResult {
  gesture: GestureType;
  confidence: number;
  stable: boolean;
  timestamp: number;
}

interface FrameData {
  timestamp: number;
  landmarks: NormalizedLandmark[][];
}

export class GestureRecognizer {
  private history: FrameData[] = [];
  private readonly historyLimit = 20; // Maintain ~0.6 seconds of history at 30fps
  
  private lastStableGesture: GestureType = 'NO_HAND';
  private gestureCounts = new Map<GestureType, number>();
  private readonly stabilityThreshold = 10; // Need 10 frames of agreement to change state
  private predictionHistory: GestureType[] = [];
  
  public processFrames(hands: NormalizedLandmark[][], timestamp: number): GestureResult {
    if (!hands || hands.length === 0) {
      this.updateHistory([], timestamp);
      return this.smoothResult('NO_HAND', 1.0, timestamp);
    }
    
    this.updateHistory(hands, timestamp);
    const { gesture, confidence } = this.classifyCurrentFrame(hands);
    
    // Only smooth if confidence is high enough, otherwise it's UNKNOWN
    const effectiveGesture = confidence > 0.65 ? gesture : 'UNKNOWN';
    return this.smoothResult(effectiveGesture, confidence, timestamp);
  }
  
  private updateHistory(hands: NormalizedLandmark[][], timestamp: number) {
    this.history.push({ timestamp, landmarks: hands });
    if (this.history.length > this.historyLimit) {
      this.history.shift();
    }
  }

  private smoothResult(rawGesture: GestureType, confidence: number, timestamp: number): GestureResult {
    this.gestureCounts.set(rawGesture, (this.gestureCounts.get(rawGesture) || 0) + 1);

    this.predictionHistory.push(rawGesture);
    if (this.predictionHistory.length > this.stabilityThreshold * 1.5) {
      this.predictionHistory.shift();
    }
    
    const counts = new Map<GestureType, number>();
    for (const g of this.predictionHistory) {
      counts.set(g, (counts.get(g) || 0) + 1);
    }
    
    let maxCount = 0;
    let candidate: GestureType = 'UNKNOWN';
    for (const [g, count] of counts.entries()) {
      if (count > maxCount) {
        maxCount = count;
        candidate = g;
      }
    }
    
    if (maxCount >= this.stabilityThreshold) {
      this.lastStableGesture = candidate;
    }
    
    return {
      gesture: this.lastStableGesture,
      confidence: confidence,
      stable: maxCount >= this.stabilityThreshold,
      timestamp
    };
  }

  private classifyCurrentFrame(hands: NormalizedLandmark[][]): { gesture: GestureType, confidence: number } {
    // Two-handed gestures
    if (hands.length === 2) {
      const hand1 = hands[0];
      const hand2 = hands[1];
      const wrist1 = hand1[0];
      const wrist2 = hand2[0];
      const wristDist = Math.sqrt((wrist1.x - wrist2.x)**2 + (wrist1.y - wrist2.y)**2);
      
      if (wristDist < 0.2) {
        const probs1 = this.getFingerProbs(hand1);
        const probs2 = this.getFingerProbs(hand2);
        const open1 = probs1.slice(1).reduce((a,b)=>a+b, 0) / 4;
        const open2 = probs2.slice(1).reduce((a,b)=>a+b, 0) / 4;
        if (open1 > 0.7 && open2 > 0.7) {
          return { gesture: 'PALMS_TOGETHER', confidence: Math.min((open1 + open2) / 2, 0.95) };
        }
      }
    }

    const hand = hands[0]; 
    const probs = this.getFingerProbs(hand);
    const wrist = hand[0];
    
    const matchStatic = (targets: number[]) => {
      let sum = 0;
      for (let i = 0; i < 5; i++) {
        sum += 1 - Math.abs(targets[i] - probs[i]);
      }
      return sum / 5.0;
    };

    const movement = this.analyzeMovement();
    
    // Distances for pinching/touching
    const dist = (p1: NormalizedLandmark, p2: NormalizedLandmark) => 
      Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2);
      
    const thumbTip = hand[4];
    const indexTip = hand[8];

    // PINCH (Thumb and Index touching, others closed)
    if (dist(thumbTip, indexTip) < 0.05 && matchStatic([0, 0, 0, 0, 0]) > 0.6) {
      return { gesture: 'PINCH', confidence: 0.9 };
    }

    // OK_SIGN (Thumb and Index touching, others extended)
    if (dist(thumbTip, indexTip) < 0.05 && matchStatic([0, 0, 1, 1, 1]) > 0.7) {
      return { gesture: 'OK_SIGN', confidence: 0.9 };
    }

    // 1. WATER: Index, Middle, Ring extended (no pinky touch)
    const waterConf = matchStatic([0, 1, 1, 1, 0]);
    if (waterConf > 0.8) return { gesture: 'WATER', confidence: waterConf };

    // 2. HELP / CALL_ME: Shaka sign (Thumb and Pinky extended)
    const helpConf = matchStatic([1, 0, 0, 0, 1]);
    if (helpConf > 0.8) {
      // If hand is high up (near ear/face), it's CALL_ME
      if (wrist.y < 0.4) return { gesture: 'CALL_ME', confidence: helpConf };
      return { gesture: 'HELP', confidence: helpConf };
    }

    // Basic Thumbs / Fist (Thumb extended, others closed)
    const fistWithThumbConf = matchStatic([1, 0, 0, 0, 0]);
    if (fistWithThumbConf > 0.8) {
      const thumbIp = hand[3];
      if (thumbTip.y < thumbIp.y && thumbTip.y < wrist.y - 0.05) {
        if (Math.abs(movement.dy) > 0.1) return { gesture: 'YES', confidence: fistWithThumbConf };
        return { gesture: 'THUMBS_UP', confidence: fistWithThumbConf };
      } else if (thumbTip.y > thumbIp.y && thumbTip.y > wrist.y + 0.05) {
        if (Math.abs(movement.dy) > 0.1) return { gesture: 'NO', confidence: fistWithThumbConf };
        return { gesture: 'THUMBS_DOWN', confidence: fistWithThumbConf };
      }
    }

    // Fist (all closed)
    const fistConf = matchStatic([0, 0, 0, 0, 0]);
    if (fistConf > 0.85) {
      return { gesture: 'FIST', confidence: fistConf };
    }

    // Pointing (Index only)
    const pointConf = matchStatic([0, 1, 0, 0, 0]);
    if (pointConf > 0.85) {
      const indexPip = hand[6];
      // Check for INDEX_HOOK
      if (dist(indexTip, wrist) < dist(indexPip, wrist)) {
        return { gesture: 'INDEX_HOOK', confidence: pointConf };
      }
      if (indexTip.y > wrist.y + 0.1) return { gesture: 'POINT_DOWN', confidence: pointConf };
      if (indexTip.x < wrist.x - 0.1) return { gesture: 'POINT_RIGHT', confidence: pointConf }; 
      if (indexTip.x > wrist.x + 0.1) return { gesture: 'POINT_LEFT', confidence: pointConf };
      return { gesture: 'POINT_UP', confidence: pointConf };
    }

    // Peace (Index and Middle)
    const peaceConf = matchStatic([0, 1, 1, 0, 0]);
    if (peaceConf > 0.85) return { gesture: 'PEACE', confidence: peaceConf };

    // Open Palm (All extended)
    const palmConf = matchStatic([1, 1, 1, 1, 1]);
    if (palmConf > 0.85) {
      if (movement.totalDistX > 0.15) {
        return { gesture: 'HELLO', confidence: palmConf }; // Waving
      } else if (movement.dy > 0.1) {
        return { gesture: 'THANK_YOU', confidence: palmConf }; // Moving down
      } else if (movement.dy < -0.1) {
        return { gesture: 'FLAT_HAND_UP', confidence: palmConf }; // Moving up/flat hand
      } else if (movement.totalDistX < 0.05 && Math.abs(movement.dy) < 0.05) {
        if (wrist.y > 0.6) return { gesture: 'HAND_ON_CHEST', confidence: palmConf };
        return { gesture: 'OPEN_PALM', confidence: palmConf }; 
      }
      return { gesture: 'OPEN_PALM', confidence: palmConf };
    }
    
    // CUPPED_HAND (all fingers half bent)
    const cuppedConf = matchStatic([0.5, 0.5, 0.5, 0.5, 0.5]);
    if (cuppedConf > 0.85) {
       return { gesture: 'CUPPED_HAND', confidence: cuppedConf };
    }

    return { gesture: 'UNKNOWN', confidence: 0 };
  }

  private getFingerProbs(landmarks: NormalizedLandmark[]): number[] {
    const dist = (p1: NormalizedLandmark, p2: NormalizedLandmark) => 
      Math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2);
      
    const wrist = landmarks[0];
    
    const getProb = (tipIdx: number, pipIdx: number) => {
      const ratio = dist(wrist, landmarks[tipIdx]) / dist(wrist, landmarks[pipIdx]);
      return Math.max(0, Math.min(1, (ratio - 0.9) / (1.4 - 0.9)));
    };
    
    const getThumbProb = () => {
      const ratio = dist(wrist, landmarks[4]) / dist(wrist, landmarks[2]);
      return Math.max(0, Math.min(1, (ratio - 0.9) / (1.3 - 0.9)));
    };

    return [
      getThumbProb(), // Thumb
      getProb(8, 6),  // Index
      getProb(12, 10),// Middle
      getProb(16, 14),// Ring
      getProb(20, 18) // Pinky
    ];
  }

  private analyzeMovement(): { dx: number, dy: number, totalDistX: number } {
    if (this.history.length < 10) return { dx: 0, dy: 0, totalDistX: 0 };
    
    const validFrames = this.history.filter(f => f.landmarks.length > 0);
    if (validFrames.length < 10) return { dx: 0, dy: 0, totalDistX: 0 };

    const oldest = validFrames[0].landmarks[0][0]; // wrist
    const newest = validFrames[validFrames.length - 1].landmarks[0][0];
    
    let totalX = 0;
    for (let i = 1; i < validFrames.length; i++) {
      totalX += Math.abs(validFrames[i].landmarks[0][0].x - validFrames[i-1].landmarks[0][0].x);
    }
    
    return {
      dx: newest.x - oldest.x,
      dy: newest.y - oldest.y,
      totalDistX: totalX
    };
  }
}
