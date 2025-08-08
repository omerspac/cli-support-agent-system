import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool
from agents.run import RunConfig
from pydantic import BaseModel
from typing import Optional

class UserContext(BaseModel):
    name: Optional[str] = None
    is_premium_user: bool = False
    issue_type: Optional[str] = None 

# ENVIRONMENT VARIABLES & CONFIGURATION

load_dotenv()

set_tracing_disabled(disabled = True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model_gemini = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model_gemini,
    model_provider=external_client,
    tracing_disabled=True
)

# FUNCTION TOOLS
@function_tool
async def refund_tool(context: UserContext):
    """
    Process a refund for a premium user.
    """
    return f"✅ Refund processed for {context.name or 'user'}."

refund_tool.is_enabled = lambda tool, context: getattr(context, "is_premium_user", False)

@function_tool
async def restart_service_tool(context: UserContext):
    """
    Restart the user's service if the issue is technical.
    """
    return "🔄 Service restarted successfully."

restart_service_tool.is_enabled = lambda tool, context: getattr(context, "issue_type", None) == "technical"


# AGENTS DEFINITIONS
triage_agent = Agent(
    name="Triage Bot",
    instructions="""
    You are a routing bot.  
    Analyze the user query and respond ONLY with one word:  
    'billing', 'technical', or 'general'.  
    Do not write anything else.
    """,
    model=model_gemini
)

billing_agent = Agent(
    name="Billing Bot",
    instructions="""
    Your work is to handle refunds and account-related issues.
    """,
    model=model_gemini,
    tools=[refund_tool]
)

tech_agent = Agent(
    name="Tech Support Bot",
    instructions="""
    Your work is to handle technical problems like restarting services.
    """,
    model=model_gemini,
    tools=[restart_service_tool]
)

general_agent = Agent(
    name="General Bot",
    instructions="""
    Your work is to handle general queries.
    """,
    model=model_gemini
)

# HANDLE QUERY
async def handle_query(prompt: str, context: UserContext):
    triage_result = await Runner.run(
        triage_agent,
        prompt,
        run_config=config,
        context=context
    )

    category = triage_result.final_output.strip().lower()
    context.issue_type = category

    if category == "billing":
        return await Runner.run(
            billing_agent,
            prompt,
            run_config=config,
            context=context
        )

    elif category == "technical":
        return await Runner.run(
            tech_agent,
            prompt,
            run_config=config,
            context=context
        )

    else:
        return await Runner.run(
            general_agent,
            prompt,
            run_config=config,
            context=context
        )


# MAIN LOOP
async def run_loop():
    user_context = UserContext(name="Omer", is_premium_user=True)

    print("AI Bot:👋 Hello! I am a console based support agent created by Muhammad Omer.")
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue

            result = await handle_query(prompt, user_context)
            print("\nAI Bot:", result.final_output)

        except KeyboardInterrupt:
            print("\nAI Bot:👋 Exiting. Thank you for using the bot!")
            break

        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_loop())