"""
=============================================================================
OCI Enterprise AI Agents — Your First Agent
=============================================================================
Companion code for the "OCI Enterprise AI Agents — Beginner Course"

This script demonstrates two examples:
  1. Simple Chat    — Call the OCI Responses API with a text prompt
  2. Function Agent — An agent that uses Function Calling to look up weather

Prerequisites:
  pip install openai python-dotenv

Authentication (choose one):
  Option A — API Key (simplest, recommended for beginners):
    1. Create an API Key in the OCI Console → Generative AI → API Keys
    2. Add an IAM policy:
       allow any-user to use generative-ai-family in compartment <name>
         where ALL { request.principal.type='generativeaiapikey' }
    3. Put your settings in a .env file next to this script (a .env in
       the course root folder, or environment variables, also work):
         OCI_GENAI_API_KEY=your-key-here
         PROJECT_ID=ocid1.generativeaiproject.oc1...   (required)
         REGION=us-chicago-1                          (optional, this is the default)
         MODEL_ID=openai.gpt-oss-120b                 (optional, this is the default)

  Option B — OCI IAM Auth (production):
    pip install oci-genai-auth
    See: https://github.com/oracle-samples/oci-genai-auth-python

Usage:
  python oci_first_agent_simple.py

Reference:
  https://docs.oracle.com/en-us/iaas/Content/generative-ai/agents.htm
  https://github.com/oracle-samples/oci-openai
=============================================================================
"""

from openai import OpenAI
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load settings from a .env next to this script, then from the course root.
# load_dotenv never overwrites a value that is already set.
script_dir = Path(__file__).resolve().parent
load_dotenv(dotenv_path=script_dir / ".env")
load_dotenv(dotenv_path=script_dir.parent / ".env")

# ─── Configuration ──────────────────────────────────────────────────────────
# IMPORTANT: Set REGION in your .env to match YOUR OCI tenancy region.
# Supported regions: us-chicago-1, us-ashburn-1, us-phoenix-1,
#                    eu-frankfurt-1, ap-hyderabad-1, ap-osaka-1

OCI_REGION = os.getenv("REGION", "us-chicago-1")

# Two endpoint paths exist — pick the one matching your auth method:
#   API Key auth  →  /openai/v1
#   oci-openai    →  /20231130/actions/v1

OCI_BASE_URL = (
    f"https://inference.generativeai.{OCI_REGION}"
    f".oci.oraclecloud.com/openai/v1"
)

# Model options (Responses API compatible):
#   "openai.gpt-oss-120b"  — OpenAI GPT-OSS (strong reasoning)
#   "xai.grok-3"           — xAI Grok 3 (fast, versatile)
#   "xai.grok-3-mini"      — xAI Grok 3 Mini (lightweight)
#   "xai.grok-4"           — xAI Grok 4 (latest)

MODEL = os.getenv("MODEL_ID", "openai.gpt-oss-120b")

# OCI Project ID (REQUIRED)
# Find it in the OCI Console → Generative AI → Projects → your project → OCID
# It looks like: ocid1.generativeaiproject.oc1.us-chicago-1.amaaaa...
OCI_PROJECT_ID = os.getenv("PROJECT_ID", "")


# ─── Create the Client ──────────────────────────────────────────────────────
def create_client():
    """Create an OpenAI client pointing to OCI Generative AI."""
    api_key = os.getenv("OCI_GENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OCI_GENAI_API_KEY in your .env file.")
        exit(1)

    if not OCI_PROJECT_ID or OCI_PROJECT_ID.startswith("<"):
        print("ERROR: Set PROJECT_ID in your .env file.")
        print("       OCI Console → Generative AI → Projects → Copy OCID")
        exit(1)

    print(f"  Endpoint : {OCI_BASE_URL}")
    print(f"  Project  : {OCI_PROJECT_ID[:40]}...")
    print("  API Key  : loaded")
    print()

    client = OpenAI(
        api_key=api_key,
        base_url=OCI_BASE_URL,
        default_headers={
            "OpenAI-Project": OCI_PROJECT_ID,
        },
    )
    return client


# =============================================================================
# EXAMPLE 1: Simple Chat
# =============================================================================
def simple_chat_example():
    """
    The simplest possible agent — a single Responses API call.
    This is identical to an OpenAI API call, just with a different
    base_url and model name.
    """
    print("=" * 60)
    print("EXAMPLE 1: Simple Chat")
    print("=" * 60)

    client = create_client()

    response = client.responses.create(
        model=MODEL,
        input="Explain what an AI agent is in 3 sentences, "
              "using a real-world analogy."
    )

    print(f"\nModel: {MODEL}")
    print(f"Response:\n{response.output_text}\n")


# =============================================================================
# EXAMPLE 2: Function Calling Agent
# =============================================================================

# Step 1: Define your local function(s)
def get_weather(city: str) -> str:
    """
    Simulated weather lookup.
    In production, replace this with a real weather API call
    (e.g., OpenWeatherMap, WeatherAPI, etc.)
    """
    weather_data = {
        "San Francisco": "65°F (18°C), Foggy with partial clearing",
        "Tokyo":         "72°F (22°C), Sunny and clear skies",
        "London":        "55°F (13°C), Overcast with light rain",
        "New York":      "78°F (26°C), Partly cloudy",
        "Sydney":        "68°F (20°C), Clear and breezy",
    }
    return weather_data.get(
        city,
        f"Weather data is not available for '{city}'. "
        f"Available cities: {', '.join(weather_data.keys())}"
    )


# Step 2: Define the tool schema (JSON Schema for the function)
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": (
        "Get the current weather conditions for a given city. "
        "Returns temperature and conditions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city name, e.g. 'San Francisco'"
            }
        },
        "required": ["city"],
    },
}


def function_calling_example():
    """
    A complete function-calling agent loop:
      1. User asks a question
      2. Model decides to call get_weather()
      3. We execute the function locally
      4. We feed the result back to the model
      5. Model generates a natural language answer
    """
    print("=" * 60)
    print("EXAMPLE 2: Function Calling Agent")
    print("=" * 60)

    client = create_client()
    user_question = "What's the weather like in San Francisco and Tokyo?"

    print(f"\nUser: {user_question}\n")

    # Step 3: Call the Responses API with the tool definition
    response = client.responses.create(
        model=MODEL,
        input=user_question,
        tools=[WEATHER_TOOL],
    )

    # Step 4: Process the response — check for function calls
    tool_results = []
    for item in response.output:
        if item.type == "function_call":
            # The model wants to call our function
            func_name = item.name
            func_args = json.loads(item.arguments)

            print(f"  Agent calls: {func_name}({func_args})")

            # Execute the function locally
            if func_name == "get_weather":
                result = get_weather(**func_args)
            else:
                result = f"Unknown function: {func_name}"

            print(f"  Result: {result}")

            # Collect the result to send back
            tool_results.append({
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": result,
            })

    # Step 5: If we have tool results, send them back to the model
    if tool_results:
        print("\n  Sending results back to model...\n")
        final_response = client.responses.create(
            model=MODEL,
            previous_response_id=response.id,
            input=tool_results,
        )
        print(f"Agent: {final_response.output_text}\n")
    else:
        # Model answered directly without needing tools
        print(f"Agent: {response.output_text}\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "─" * 60)
    print("  OCI Enterprise AI Agents — First Agent Demo")
    print("─" * 60 + "\n")

    # Run Example 1: Simple Chat
    simple_chat_example()

    print()

    # Run Example 2: Function Calling Agent
    function_calling_example()

    print("─" * 60)
    print("  Done! You've built your first OCI Enterprise AI Agent.")
    print("─" * 60)
    print()
    print("Next steps:")
    print("  1. Try changing the MODEL to 'xai.grok-3'")
    print("  2. Add more tools (e.g., a calculator, a database lookup)")
    print("  3. Use the Conversations API for multi-turn chat")
    print("  4. Deploy your agent as a Hosted Application")
    print()
    print("Docs: https://docs.oracle.com/en-us/iaas/Content/"
          "generative-ai/agents.htm")

