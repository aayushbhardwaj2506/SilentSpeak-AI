import os
import json
from pydantic import BaseModel, Field
from google.adk.agents import LlmAgent
from typing import Optional, Literal

# Input Schema
class GestureEvent(BaseModel):
    gesture: str
    confidence: float
    stable: bool
    timestamp: int

class ActionObservation(BaseModel):
    duration: float
    observations: list[str]

# Output Schema
class AgentDecision(BaseModel):
    intent: str = Field(description="The semantic intent of the gesture (e.g. REQUEST_HELP)")
    decision: Literal["SPEAK", "IGNORE", "CONFIRM"] = Field(description="Must be exactly one of: SPEAK, IGNORE, CONFIRM")
    response_text: str = Field(description="The natural language response or text to be spoken/displayed")
    confidence: float = Field(description="The confidence level in this interpretation (0.0 to 1.0)")

class SilentSpeakAgent:
    def __init__(self):
        # We fetch the model from env, default to gemini-3.6-flash
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        
        # We need a system instruction to guide the agent
        self.instruction = """
        You are the communication reasoning agent for SilentSpeak AI. 
        You receive structured gesture-recognition events and determine the user's likely communication intent.
        
        Controlled Gesture Vocabulary Map:
        - HELLO -> GREETING
        - YES -> AFFIRMATION
        - NO -> NEGATION
        - HELP -> REQUEST_HELP
        - WATER -> REQUEST_WATER
        - STOP -> STOP_ACTION
        - THANK_YOU -> GRATITUDE
        - UNKNOWN -> UNKNOWN
        - NO_HAND -> NO_INPUT

        Rules:
        1. Consider the provided confidence score. 
           - High confidence (>0.7): interpret normally. Set decision to "SPEAK".
           - Low confidence (<=0.7): respond cautiously. Set decision to "CONFIRM" or "IGNORE".
        2. UNKNOWN: Do not guess. Respond with intent "UNKNOWN" and decision "IGNORE".
        3. NO_HAND: Do not generate a communication action.
        4. Return your output EXACTLY matching the provided structured format.
        5. The 'decision' field MUST be exactly one of: "SPEAK", "IGNORE", or "CONFIRM". DO NOT use any other string.
        """

        # Initialize the official Google ADK Agent
        self.agent = LlmAgent(
            name="silentspeak_communication_agent",
            model=model_name,
            instruction=self.instruction,
            # For pydantic output parsing (if supported directly in adk wrapper, else we can parse the string)
            # Many ADK wrappers allow providing response_format or we just ask for JSON
        )
        
        from google.adk import Runner
        from google.adk.sessions import InMemorySessionService
        self.runner = Runner(
            agent=self.agent,
            app_name="silentspeak_communication_agent",
            session_service=InMemorySessionService(),
            auto_create_session=True
        )

    def interpret(self, event: GestureEvent, context: Optional[list] = None) -> AgentDecision:
        # Pre-filter
        if event.gesture == "NO_HAND":
            return AgentDecision(
                intent="NO_INPUT",
                decision="IGNORE",
                response_text="",
                confidence=1.0
            )
            
        # Build prompt
        prompt = f"Current Gesture Event:\n"
        prompt += f"- Gesture: {event.gesture}\n"
        prompt += f"- Confidence: {event.confidence}\n"
        prompt += f"- Stable: {event.stable}\n"
        
        if context and len(context) > 0:
            prompt += f"\nRecent Context:\n{json.dumps(context, indent=2)}\n"
            
        prompt += "\nPlease return a JSON object containing keys: intent, decision, response_text, and confidence."

        # Call the ADK Agent using Runner
        from google.genai import types
        msg = types.Content(role='user', parts=[types.Part.from_text(text=prompt)])
        
        response_text = ""
        for ev in self.runner.run(user_id="1", session_id="1", new_message=msg):
            if hasattr(ev, 'content') and ev.content and hasattr(ev.content, 'parts') and ev.content.parts:
                for part in ev.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
        
        # Extract text
        text = response_text
        
        # Clean markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
            return AgentDecision(
                intent=data.get("intent", "UNKNOWN"),
                decision=data.get("decision", "IGNORE"),
                response_text=data.get("response_text", "I did not understand."),
                confidence=float(data.get("confidence", event.confidence))
            )
        except Exception as e:
            # Fallback for parsing errors
            print("Failed to parse agent response:", text)
            return AgentDecision(
                intent="ERROR",
                decision="IGNORE",
                response_text="Error processing intent.",
                confidence=0.0
            )

    def interpret_with_groq(self, event: GestureEvent, context: Optional[list] = None) -> AgentDecision:
        import requests
        
        # Pre-filter
        if event.gesture == "NO_HAND":
            return AgentDecision(
                intent="NO_INPUT", decision="IGNORE", response_text="", confidence=1.0
            )
            
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise Exception("GROQ_API_KEY is not set in environment.")
            
        prompt = f"Current Gesture Event:\n- Gesture: {event.gesture}\n- Confidence: {event.confidence}\n- Stable: {event.stable}\n"
        if context and len(context) > 0:
            prompt += f"\nRecent Context:\n{json.dumps(context, indent=2)}\n"
        prompt += "\nPlease return a JSON object containing keys: intent, decision, response_text, and confidence."

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            
            return AgentDecision(
                intent=data.get("intent", "UNKNOWN"),
                decision=data.get("decision", "IGNORE"),
                response_text=data.get("response_text", "I did not understand."),
                confidence=float(data.get("confidence", event.confidence))
            )
        except Exception as e:
            print("Groq Error:", str(e))
            if 'resp' in locals() and hasattr(resp, 'text'):
                print("Response:", resp.text)
            return AgentDecision(
                intent="ERROR",
                decision="IGNORE",
                response_text="Error processing intent.",
                confidence=0.0
            )

    def interpret_observation_with_groq(self, event: ActionObservation) -> AgentDecision:
        import requests
        
        if not event.observations or len(event.observations) == 0:
             return AgentDecision(intent="NO_INPUT", decision="IGNORE", response_text="", confidence=1.0)
             
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise Exception("GROQ_API_KEY is not set in environment.")
            
        system_instruction = """
        You are the Action Observation semantic interpreter for SilentSpeak AI.
        You receive a list of physical observations that occurred over a short time window.
        Your goal is to infer the natural-language sentence the user intends to communicate.
        
        Rules:
        1. Treat observations as physical/action descriptions, NOT predefined gesture meanings. A "significant waving motion" must NEVER automatically map to NEGATION.
        2. Infer communicative intent from the complete sequence of observations.
        3. Use broader intents such as: GREETING, ATTENTION, REQUEST, QUESTION, DIRECTION, FOOD_REQUEST, DRINK_REQUEST, AGREEMENT, DISAGREEMENT, EMOTION, INFORMATION, UNKNOWN.
        4. Do not force an observation into an unrelated intent. If one physical action is ambiguous, return a sensible UNKNOWN/AMBIGUOUS interpretation rather than hallucinating.
        5. Do not produce generic responses such as "Understood" unless the user actually communicated acknowledgement.
        6. Set 'decision' to "SPEAK" if you confidently infer an intent, otherwise "IGNORE" or "CONFIRM".
        7. Your 'response_text' MUST be a natural sentence (e.g., "Please come over here.", "I would like some water.").
        8. Return your output EXACTLY matching the JSON schema containing: intent, decision, response_text, and confidence.
        """
        
        prompt = f"Observation Window (Duration: {event.duration}s):\n"
        for i, obs in enumerate(event.observations):
            prompt += f"{i+1}. {obs}\n"
        prompt += "\nPlease return a JSON object containing keys: intent, decision, response_text, and confidence."

        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "qwen/qwen3.8-27b",
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        try:
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            
            return AgentDecision(
                intent=data.get("intent", "UNKNOWN"),
                decision=data.get("decision", "IGNORE"),
                response_text=data.get("response_text", "I did not understand."),
                confidence=float(data.get("confidence", 0.9))
            )
        except Exception as e:
            print("Groq Observation Error:", str(e))
            if 'resp' in locals() and hasattr(resp, 'text'):
                print("Response:", resp.text)
            return AgentDecision(
                intent="ERROR",
                decision="IGNORE",
                response_text="Error processing observation.",
                confidence=0.0
            )

# Singleton instance
silent_speak_agent = SilentSpeakAgent()
