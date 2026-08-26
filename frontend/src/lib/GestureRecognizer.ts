export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}
export type GestureType = 'HELLO' | 'YES' | 'NO' | 'HELP' | 'WATER' | 'STOP' | 'THANK_YOU' | 'UNKNOWN' | 'NO_HAND' | 
  'NUMBER_0' | 'NUMBER_1' | 'NUMBER_2' | 'NUMBER_3' | 'NUMBER_4' | 'NUMBER_5' | 
  'THUMBS_UP' | 'THUMBS_DOWN' | 'PEACE' | 'OK_SIGN' | 'FIST' | 'OPEN_PALM' |
  'POINT_UP' | 'POINT_DOWN' | 'POINT_LEFT' | 'POINT_RIGHT';

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
    const hand = hands[0]; 
    const probs = this.getFingerProbs(hand);
    
    const matchStatic = (targets: number[]) => {
      let sum = 0;
      for (let i = 0; i < 5; i++) {
        sum += 1 - Math.abs(targets[i] - probs[i]);
      }
      return sum / 5.0;
    };

    const movement = this.analyzeMovement();

    // 1. WATER: Index, Middle, Ring extended
    const waterConf = matchStatic([0, 1, 1, 1, 0]);
    if (waterConf > 0.8) return { gesture: 'WATER', confidence: waterConf };

    // 2. HELP: Shaka sign (Thumb and Pinky extended)
    const helpConf = matchStatic([1, 0, 0, 0, 1]);
    if (helpConf > 0.8) return { gesture: 'HELP', confidence: helpConf };

    // Basic Thumbs / Fist (Thumb extended, others closed)
    const fistWithThumbConf = matchStatic([1, 0, 0, 0, 0]);
    if (fistWithThumbConf > 0.8) {
      const wrist = hand[0];
      const thumbTip = hand[4];
      const thumbIp = hand[3];
      // Y increases downwards
      if (thumbTip.y < thumbIp.y && thumbTip.y < wrist.y - 0.05) {
        if (Math.abs(movement.dy) > 0.1) return { gesture: 'YES', confidence: fistWithThumbConf };
        return { gesture: 'THUMBS_UP', confidence: fistWithThumbConf };
      } else if (thumbTip.y > thumbIp.y && thumbTip.y > wrist.y + 0.05) {
        if (Math.abs(movement.dy) > 0.1) return { gesture: 'NO', confidence: fistWithThumbConf };
        return { gesture: 'THUMBS_DOWN', confidence: fistWithThumbConf };
      }
    }

    // Numbers 0-5
    const num0Conf = matchStatic([0, 0, 0, 0, 0]); // FIST
    if (num0Conf > 0.85) return { gesture: 'NUMBER_0', confidence: num0Conf };

    const num1Conf = matchStatic([0, 1, 0, 0, 0]); // POINTING
    if (num1Conf > 0.85) {
      const wrist = hand[0];
      const tip = hand[8];
      if (tip.y < wrist.y - 0.1) return { gesture: 'NUMBER_1', confidence: num1Conf }; // pointing up is 1
      if (tip.y > wrist.y + 0.1) return { gesture: 'POINT_DOWN', confidence: num1Conf };
      if (tip.x < wrist.x - 0.1) return { gesture: 'POINT_RIGHT', confidence: num1Conf }; 
      if (tip.x > wrist.x + 0.1) return { gesture: 'POINT_LEFT', confidence: num1Conf };
      return { gesture: 'POINT_UP', confidence: num1Conf };
    }

    const num2Conf = matchStatic([0, 1, 1, 0, 0]); // PEACE / 2
    if (num2Conf > 0.85) return { gesture: 'NUMBER_2', confidence: num2Conf };

    const num3Conf = matchStatic([1, 1, 1, 0, 0]); // Standard 3
    if (num3Conf > 0.85) return { gesture: 'NUMBER_3', confidence: num3Conf };

    const num4Conf = matchStatic([0, 1, 1, 1, 1]);
    if (num4Conf > 0.85) return { gesture: 'NUMBER_4', confidence: num4Conf };

    const num5Conf = matchStatic([1, 1, 1, 1, 1]);
    if (num5Conf > 0.85) {
      if (movement.totalDistX > 0.15) {
        return { gesture: 'HELLO', confidence: num5Conf }; // Waving
      } else if (movement.dy > 0.1) {
        return { gesture: 'THANK_YOU', confidence: num5Conf }; // Moving down
      } else if (movement.totalDistX < 0.05 && Math.abs(movement.dy) < 0.05) {
        return { gesture: 'NUMBER_5', confidence: num5Conf }; 
      }
      return { gesture: 'OPEN_PALM', confidence: num5Conf };
    }
    
    // OK sign
    const okConf = matchStatic([0, 0, 1, 1, 1]);
    if (okConf > 0.8) {
      const thumbTip = hand[4];
      const indexTip = hand[8];
      const dist = Math.sqrt((thumbTip.x - indexTip.x)**2 + (thumbTip.y - indexTip.y)**2);
      if (dist < 0.05) return { gesture: 'OK_SIGN', confidence: okConf };
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
