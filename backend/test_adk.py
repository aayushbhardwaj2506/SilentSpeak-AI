from google.adk.agents import LlmAgent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import os
from dotenv import load_dotenv
load_dotenv()
agent = LlmAgent(name='test_agent', model='gemini-3.6-flash')
runner = Runner(agent=agent, app_name='test_app', session_service=InMemorySessionService(), auto_create_session=True)
msg = types.Content(role='user', parts=[types.Part.from_text(text='say hello')])

for event in runner.run(user_id='1', session_id='1', new_message=msg):
    print(repr(event))
    print(repr(event.output))
    if hasattr(event.output, 'text'):
        print('TEXT:', event.output.text)
