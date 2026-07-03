import asyncio
from lib.redis.client import get_redis_client
from graphs.shopping_graph import get_shopping_graph, ShoppingGraphDeps
from graphs.state import AgentState
from lib.chat.deps import resolve_turn_state

async def main():
    redis = get_redis_client()
    graph = await get_shopping_graph(redis)
    
    # turn 1: weather
    print("Turn 1...")
    state1 = await resolve_turn_state(
        graph,
        message="What's the weather in Colombo?",
        session_id="e2e_test_123",
        zep_thread_id="e2e_test_123",
        config={"configurable": {"thread_id": "e2e_test_123"}}
    )
    
    config = {"configurable": {"thread_id": "e2e_test_123"}}
    async for chunk in graph.astream(state1, config, stream_mode=["updates"]):
        print(chunk)
        
    print("\nTurn 2...")
    state2 = await resolve_turn_state(
        graph,
        message="Can you deliver a live elephant?",
        session_id="e2e_test_123",
        zep_thread_id="e2e_test_123",
        config={"configurable": {"thread_id": "e2e_test_123"}}
    )
    async for chunk in graph.astream(state2, config, stream_mode=["updates"]):
        print(chunk)

asyncio.run(main())
