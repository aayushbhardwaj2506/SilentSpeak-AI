export type PriorityLevel = 'NORMAL' | 'HIGH' | 'EMERGENCY';

export interface GestureDefinition {
  token: string;
  category: string;
  displayName: string;
  possibleMeanings: string[];
  directSpeech?: string; // If defined, bypasses Groq (e.g., "Five.")
  requiresContext: boolean;
  priority: PriorityLevel;
}

// Registry mapping semantic concepts to vocabulary rules
export const GESTURE_VOCABULARY: Record<string, GestureDefinition> = {


  // =========================================================================
  // 2. CORE CONFIRMATION & SOCIAL
  // =========================================================================
  YES: { token: 'YES', category: 'confirmation', displayName: 'Yes', possibleMeanings: ['yes', 'agree', 'correct', 'understood'], requiresContext: true, priority: 'NORMAL' },
  NO: { token: 'NO', category: 'confirmation', displayName: 'No', possibleMeanings: ['no', 'disagree', 'incorrect', 'reject'], requiresContext: true, priority: 'NORMAL' },
  THUMBS_UP: { token: 'THUMBS_UP', category: 'confirmation', displayName: 'Thumbs Up', possibleMeanings: ['yes', 'okay', 'good', 'agree', 'done', 'understood'], requiresContext: true, priority: 'NORMAL' },
  THUMBS_DOWN: { token: 'THUMBS_DOWN', category: 'confirmation', displayName: 'Thumbs Down', possibleMeanings: ['no', 'bad', 'disagree', 'not done', 'sad', 'dislike'], requiresContext: true, priority: 'NORMAL' },
  OK_SIGN: { token: 'OK_SIGN', category: 'confirmation', displayName: 'Okay', possibleMeanings: ['okay', 'fine', 'perfect', 'number nine'], requiresContext: true, priority: 'NORMAL' },
  OPEN_PALM: { token: 'OPEN_PALM', category: 'action', displayName: 'Open Palm', possibleMeanings: ['stop', 'wait', 'hello', 'show me', 'attention', 'five'], requiresContext: true, priority: 'HIGH' },
  HELLO: { token: 'HELLO', category: 'social', displayName: 'Hello', possibleMeanings: ['hello', 'hi', 'greetings', 'bye', 'goodbye', 'wave'], requiresContext: true, priority: 'NORMAL' },
  THANK_YOU: { token: 'THANK_YOU', category: 'social', displayName: 'Thank You', possibleMeanings: ['thank you', 'thanks', 'gratitude'], requiresContext: true, priority: 'NORMAL' },
  PALMS_TOGETHER: { token: 'PALMS_TOGETHER', category: 'social', displayName: 'Palms Together', possibleMeanings: ['namaste', 'please', 'sorry', 'respect', 'welcome', 'thank you'], requiresContext: true, priority: 'NORMAL' },

  // =========================================================================
  // 3. DIRECTIONS / SPATIAL
  // =========================================================================
  POINT_UP: { token: 'POINT_UP', category: 'spatial', displayName: 'Point Up', possibleMeanings: ['up', 'above', 'top', 'high', 'look up'], requiresContext: true, priority: 'NORMAL' },
  POINT_DOWN: { token: 'POINT_DOWN', category: 'spatial', displayName: 'Point Down', possibleMeanings: ['down', 'below', 'bottom', 'here', 'look down'], requiresContext: true, priority: 'NORMAL' },
  POINT_LEFT: { token: 'POINT_LEFT', category: 'spatial', displayName: 'Point Left', possibleMeanings: ['left', 'there', 'that', 'look left'], requiresContext: true, priority: 'NORMAL' },
  POINT_RIGHT: { token: 'POINT_RIGHT', category: 'spatial', displayName: 'Point Right', possibleMeanings: ['right', 'there', 'that', 'look right'], requiresContext: true, priority: 'NORMAL' },

  // =========================================================================
  // 4. BASIC COMMUNICATION & ACTIONS
  // =========================================================================
  FIST: { token: 'FIST', category: 'action', displayName: 'Fist', possibleMeanings: ['fist', 'strong', 'angry', 'hold', 'zero', 'solid'], requiresContext: true, priority: 'NORMAL' },
  PEACE: { token: 'PEACE', category: 'action', displayName: 'Peace/Two', possibleMeanings: ['peace', 'victory', 'two'], requiresContext: true, priority: 'NORMAL' },
  PINCH: { token: 'PINCH', category: 'action', displayName: 'Pinch', possibleMeanings: ['a little', 'small', 'less', 'wait a bit', 'tiny'], requiresContext: true, priority: 'NORMAL' },
  INDEX_HOOK: { token: 'INDEX_HOOK', category: 'action', displayName: 'Hook', possibleMeanings: ['who', 'what', 'question', 'confused', 'why'], requiresContext: true, priority: 'NORMAL' },
  FLAT_HAND_UP: { token: 'FLAT_HAND_UP', category: 'action', displayName: 'Flat Hand Up', possibleMeanings: ['give', 'take', 'show', 'here', 'food', 'plate'], requiresContext: true, priority: 'NORMAL' },
  HAND_ON_CHEST: { token: 'HAND_ON_CHEST', category: 'pronoun', displayName: 'Hand on Chest', possibleMeanings: ['me', 'I', 'my', 'mine', 'heart', 'feeling'], requiresContext: true, priority: 'NORMAL' },

  // =========================================================================
  // 5. FOOD / DAILY NEEDS
  // =========================================================================
  WATER: { token: 'WATER', category: 'need', displayName: 'Water', possibleMeanings: ['water', 'drink', 'thirsty', 'w'], requiresContext: true, priority: 'NORMAL' },
  CUPPED_HAND: { token: 'CUPPED_HAND', category: 'need', displayName: 'Cupped Hand', possibleMeanings: ['drink', 'water', 'tea', 'coffee', 'milk', 'medicine', 'liquid'], requiresContext: true, priority: 'NORMAL' },

  // =========================================================================
  // 6. EMERGENCY / PRIORITY
  // =========================================================================
  HELP: { token: 'HELP', category: 'emergency', displayName: 'Help', possibleMeanings: ['help', 'danger', 'emergency', 'safe', 'unsafe'], directSpeech: 'I need help.', requiresContext: false, priority: 'EMERGENCY' },
  CALL_ME: { token: 'CALL_ME', category: 'emergency', displayName: 'Call', possibleMeanings: ['call', 'phone', 'call family', 'call doctor', 'call police'], requiresContext: true, priority: 'HIGH' },
  STOP: { token: 'STOP', category: 'emergency', displayName: 'Stop', possibleMeanings: ['stop', 'emergency stop', 'no'], directSpeech: 'Stop.', requiresContext: false, priority: 'HIGH' },
};
