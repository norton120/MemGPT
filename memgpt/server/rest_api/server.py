import json
from contextlib import asynccontextmanager

from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

from memgpt.server.rest_api.agents.index import setup_agents_index_router
from memgpt.server.rest_api.agents.command import setup_agents_command_router
from memgpt.server.rest_api.agents.config import setup_agents_config_router
from memgpt.server.rest_api.agents.memory import setup_agents_memory_router
from memgpt.server.rest_api.agents.message import setup_agents_message_router
from memgpt.server.rest_api.config.index import setup_config_index_router
from memgpt.constants import JSON_ENSURE_ASCII
from memgpt.server.server import SyncServer
from memgpt.server.rest_api.interface import QueuingInterface
from memgpt.server.rest_api.static_files import mount_static_files

"""
Basic REST API sitting on top of the internal MemGPT python server (SyncServer)

Start the server with:
  cd memgpt/server/rest_api
  poetry run uvicorn server:app --reload
"""

interface: QueuingInterface = QueuingInterface()
server: SyncServer = SyncServer(default_interface=interface)


API_PREFIX = "/api"

CORS_ORIGINS = [
    "http://localhost:4200",
    "http://localhost:4201",
    "http://localhost:8283",
    "http://127.0.0.1:4200",
    "http://127.0.0.1:4201",
    "http://127.0.0.1:8283",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# /api/agents endpoints
app.include_router(setup_agents_command_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_config_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_index_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_memory_router(server, interface), prefix=API_PREFIX)
app.include_router(setup_agents_message_router(server, interface), prefix=API_PREFIX)
# /api/config endpoints
app.include_router(setup_config_index_router(server, interface), prefix=API_PREFIX)
# / static files
mount_static_files(app)


@app.on_event("startup")
def on_startup():
    # Update the OpenAPI schema
    if not app.openapi_schema:
        app.openapi_schema = app.openapi()

    if app.openapi_schema:
        app.openapi_schema["servers"] = [{"url": "http://localhost:8283"}]
        app.openapi_schema["info"]["title"] = "MemGPT API"

    # Write out the OpenAPI schema to a file
    with open("openapi.json", "w") as file:
        print(f"Writing out openapi.json file")
        json.dump(app.openapi_schema, file, indent=2)


@app.on_event("shutdown")
def on_shutdown():
    global server
    server.save_agents()
    server = None


app = FastAPI(lifespan=lifespan)

# app = FastAPI()
# server = SyncServer(default_interface=interface)


# server.list_agents
@app.get("/agents")
def list_agents(user_id: str):
    interface.clear()
    return server.list_agents(user_id=user_id)


@app.get("/agents/memory")
def get_agent_memory(user_id: str, agent_id: str):
    interface.clear()
    return server.get_agent_memory(user_id=user_id, agent_id=agent_id)


@app.put("/agents/memory")
def put_agent_memory(body: CoreMemory):
    interface.clear()
    new_memory_contents = {"persona": body.persona, "human": body.human}
    return server.update_agent_core_memory(user_id=body.user_id, agent_id=body.agent_id, new_memory_contents=new_memory_contents)


@app.get("/agents/config")
def get_agent_config(user_id: str, agent_id: str):
    interface.clear()
    return server.get_agent_config(user_id=user_id, agent_id=agent_id)


@app.get("/config")
def get_server_config(user_id: str):
    interface.clear()
    return server.get_server_config(user_id=user_id)


# server.create_agent
@app.post("/agents")
def create_agents(body: CreateAgentConfig):
    interface.clear()
    try:
        agent_id = server.create_agent(user_id=body.user_id, agent_config=body.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    return {"agent_id": agent_id}


# server.user_message
@app.post("/agents/message")
async def user_message(body: UserMessage):
    if body.stream:
        # For streaming response
        try:
            # Start the generation process (similar to the non-streaming case)
            # This should be a non-blocking call or run in a background task

            # Check if server.user_message is an async function
            if asyncio.iscoroutinefunction(server.user_message):
                # Start the async task
                asyncio.create_task(server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message))
            else:
                # Run the synchronous function in a thread pool
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, server.user_message, body.user_id, body.agent_id, body.message)

            async def formatted_message_generator():
                async for message in interface.message_generator():
                    formatted_message = f"data: {json.dumps(message, ensure_ascii=JSON_ENSURE_ASCII)}\n\n"
                    yield formatted_message
                    await asyncio.sleep(1)

            # Return the streaming response using the generator
            return StreamingResponse(formatted_message_generator(), media_type="text/event-stream")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")

    else:
        interface.clear()
        try:
            server.user_message(user_id=body.user_id, agent_id=body.agent_id, message=body.message)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{e}")
        return {"messages": interface.to_list()}


# server.run_command
@app.post("/agents/command")
def run_command(body: Command):
    interface.clear()
    try:
        response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}")
    response = server.run_command(user_id=body.user_id, agent_id=body.agent_id, command=body.command)
    return {"response": response}
