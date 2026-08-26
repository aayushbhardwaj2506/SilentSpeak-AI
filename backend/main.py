from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import os

# Load environment variables
load_dotenv()

# Import our ADK Agent
from agent import silent_speak_agent, GestureEvent, AgentDecision, ActionObservation
app = FastAPI(title="SilentSpeak AI API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to SilentSpeak AI Backend (UPDATED)!"}

@app.post("/api/agent/interpret", response_model=AgentDecision)
def interpret_gesture(event: GestureEvent):
    # Ensure API key is present
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="Gemini API Key is missing in environment variables.")
        
    try:
        # We can pass context here if the frontend provides it in the future
        decision = silent_speak_agent.interpret_with_groq(event)
        return decision
    except Exception as e:
        print(f"Agent Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process gesture with ADK Agent.")

@app.post("/api/agent/observe", response_model=AgentDecision)
def observe_action(event: ActionObservation):
    try:
        decision = silent_speak_agent.interpret_observation_with_groq(event)
        return decision
    except Exception as e:
        print(f"Observe Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process observation.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
