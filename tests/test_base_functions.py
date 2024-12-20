import pytest

import letta.functions.function_sets.base as base_functions
from letta import create_client
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.llm_config import LLMConfig

from .utils import wipe_config

# test_agent_id = "test_agent"
client = None


@pytest.fixture(scope="module")
def agent_obj():
    """Create a test agent that we can call functions on"""
    wipe_config()
    global client
    client = create_client()
    client.set_default_llm_config(LLMConfig.default_config("gpt-4o-mini"))
    client.set_default_embedding_config(EmbeddingConfig.default_config(provider="openai"))

    agent_state = client.create_agent()

    global agent_obj
    agent_obj = client.server.load_agent(agent_id=agent_state.id)
    yield agent_obj

    client.delete_agent(agent_obj.agent_state.id)


def test_archival(agent_obj):
    base_functions.archival_memory_insert(agent_obj, "banana")
    base_functions.archival_memory_search(agent_obj, "banana")
    base_functions.archival_memory_search(agent_obj, "banana", page=0)


def test_recall(agent_obj):
    base_functions.conversation_search(agent_obj, "banana")
    base_functions.conversation_search(agent_obj, "banana", page=0)
    base_functions.conversation_search_date(agent_obj, start_date="2022-01-01", end_date="2022-01-02")
