# ===================================================================================
# Project: ChatTensorflow
# File: src/services/rag/rag_service.py
# Description: RAG service functions for executing graph and managing conversations
# Author: LALAN KUMAR
# Created: [16-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

import os
import sys
from typing import Dict, Any, AsyncGenerator, List, Optional
from langchain_core.messages import HumanMessage, BaseMessage

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.logger import logging
from src.services.rag.tf_graph import create_assistant_graph
from src.services.rag.states import AgentState


# Initialize graph once at module level for reuse
_graph_instance = None


def get_graph():
    """Get or create the assistant graph instance."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = create_assistant_graph()
        logging.info("TFAssistantGraph initialized successfully")
    return _graph_instance


def _prepare_config(
    thread_id: str,
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare configuration for graph execution.
    
    Args:
        thread_id: Conversation thread ID
        user_id: User identifier
        metadata: Additional metadata for the execution
        
    Returns:
        Configuration dictionary with checkpoint settings
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": user_id
        },
        "metadata": {
            "thread_id": thread_id,
            "user_id": user_id,
            "service": "tensorflow-assistant",
            **(metadata or {})
        },
        "tags": ["tensorflow-assistant", "rag", "langgraph", f"user:{user_id}"],
        "run_name": f"tensorflow-assistant-{thread_id}"
    }
    
    return config


async def execute_rag(
    user_query: str,
    user_id: str,
    thread_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute RAG pipeline synchronously and return the complete response.
    
    Uses thread_id for checkpointing to maintain conversation state across calls.
    
    Args:
        user_query: The user's question or query
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        metadata: Additional metadata to attach to the execution
        
    Returns:
        Dictionary containing:
            - response: The final AI response text
            - metadata: Additional information (sources, research steps, etc.)
            - status: Execution status
            
    Example:
        >>> result = await execute_rag(
        ...     "How to build a CNN model?",
        ...     "user123",
        ...     "thread456"
        ... )
        >>> print(result["response"])
    """
    try:
        logging.info(f"Executing RAG for user_id={user_id}, thread_id={thread_id}")
        logging.info(f"Query: {user_query}")
        
        graph = get_graph()
        
        # Prepare configuration
        config = _prepare_config(
            thread_id=thread_id,
            user_id=user_id,
            metadata={
                "message_length": len(user_query),
                **(metadata or {})
            }
        )
        
        # Prepare input state
        input_state = AgentState(
            messages=[HumanMessage(content=user_query)]
        )
        
        # Execute the graph
        result = await graph.ainvoke(input_state, config)
        
        # Extract response and metadata
        final_message = result["messages"][-1]
        response_text = final_message.content
        
        # Build metadata
        result_metadata = {
            "router_type": result.get("router", {}).get("type"),
            "router_logic": result.get("router", {}).get("logic"),
            "research_steps": result.get("steps", []),
            "documents_count": len(result.get("documents", [])),
            "sources": [
                doc.metadata.get("source_url", "")
                for doc in result.get("documents", [])
                if hasattr(doc, "metadata")
            ],
            "summary": result.get("summary"),
            "message_count": len(result["messages"])
        }
        
        logging.info(f"RAG execution completed successfully for thread_id={thread_id}")
        
        return {
            "response": response_text,
            "metadata": result_metadata,
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error in execute_rag for thread_id={thread_id}: {e}", exc_info=True)
        return {
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "metadata": {"error": str(e)},
            "status": "error"
        }


async def execute_rag_stream(
    user_query: str,
    user_id: str,
    thread_id: str,
    stream_mode: str = "updates",
    metadata: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Execute RAG pipeline with streaming response.
    
    Yields intermediate state updates as the graph processes the query.
    Uses thread_id for checkpointing to maintain conversation state.
    
    Args:
        user_query: The user's question or query
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        stream_mode: Streaming mode - "updates" (default) or "values"
            - "updates": Yields only changed values for each node
            - "values": Yields full state after each node completes
        metadata: Additional metadata to attach to the execution
        
    Yields:
        Dictionary chunks containing:
            - type: Event type ("start", "node", "response_chunk", "end", "error")
            - data: Event-specific data
            - node: Current node being executed (for node events)
            
    Example:
        >>> async for chunk in execute_rag_stream("How to use tf.keras?", "user123", "thread456"):
        ...     if chunk["type"] == "response_chunk":
        ...         print(chunk["data"], end="", flush=True)
    """
    try:
        logging.info(f"Starting streaming RAG for user_id={user_id}, thread_id={thread_id}")
        logging.info(f"Query: {user_query}")
        
        graph = get_graph()
        
        # Prepare configuration
        config = _prepare_config(
            thread_id=thread_id,
            user_id=user_id,
            metadata={
                "message_length": len(user_query),
                "stream_mode": stream_mode,
                **(metadata or {})
            }
        )
        
        # Prepare input state
        input_state = AgentState(
            messages=[HumanMessage(content=user_query)]
        )
        
        yield {
            "type": "start",
            "data": {"query": user_query, "thread_id": thread_id},
            "node": None
        }
        
        # Stream the graph execution
        async for event in graph.astream(input_state, config, stream_mode=stream_mode):
            # Extract node name and state updates
            for node_name, node_state in event.items():
                
                # Yield node execution events
                yield {
                    "type": "node",
                    "data": {
                        "node_name": node_name,
                        "router_type": node_state.get("router", {}).get("type") if "router" in node_state else None,
                        "steps_remaining": len(node_state.get("steps", [])),
                        "documents_count": len(node_state.get("documents", []))
                    },
                    "node": node_name
                }
                
                # If we have messages, check for new AI response
                if "messages" in node_state and node_state["messages"]:
                    last_message = node_state["messages"][-1]
                    if hasattr(last_message, "content") and last_message.content:
                        yield {
                            "type": "response_chunk",
                            "data": last_message.content,
                            "node": node_name
                        }
        
        # Get final state for metadata
        final_state = await graph.aget_state(config)
        
        metadata_result = {
            "router_type": final_state.values.get("router", {}).get("type"),
            "research_steps": final_state.values.get("steps", []),
            "documents_count": len(final_state.values.get("documents", [])),
            "sources": [
                doc.metadata.get("source_url", "")
                for doc in final_state.values.get("documents", [])
                if hasattr(doc, "metadata")
            ],
            "summary": final_state.values.get("summary"),
            "message_count": len(final_state.values["messages"])
        }
        
        yield {
            "type": "end",
            "data": {"metadata": metadata_result, "status": "success"},
            "node": None
        }
        
        logging.info(f"Streaming RAG completed for thread_id={thread_id}")
        
    except Exception as e:
        logging.error(f"Error in execute_rag_stream for thread_id={thread_id}: {e}", exc_info=True)
        
        # Provide more specific error messages
        error_message = str(e)
        if "single turn requests end with a user role" in error_message:
            error_message = "There was an issue with the conversation format. Please try starting a new conversation."
        elif "Invalid argument provided to Gemini" in error_message:
            error_message = "The AI service encountered an error. Please try rephrasing your question or start a new conversation."
        
        yield {
            "type": "error",
            "data": {"error": error_message},
            "node": None
        }


async def get_conversation_history(
    user_id: str,
    thread_id: str,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Retrieve conversation history for a specific thread.
    
    Uses the checkpointer to load the last state of the conversation.
    
    Args:
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        limit: Optional limit on number of messages to return (most recent)
        
    Returns:
        Dictionary containing:
            - messages: List of conversation messages
            - summary: Current conversation summary (if available)
            - metadata: Additional thread information
            - status: Retrieval status ("success", "not_found", "error")
            
    Example:
        >>> history = await get_conversation_history("user123", "thread456", limit=10)
        >>> for msg in history["messages"]:
        ...     print(f"{msg['role']}: {msg['content']}")
    """
    try:
        logging.info(f"Retrieving conversation history for user_id={user_id}, thread_id={thread_id}")
        
        graph = get_graph()
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Get current state
        state = await graph.aget_state(config)
        
        # Check if state exists and has values
        if not state or not state.values:
            logging.info(f"No conversation history found for thread_id={thread_id}")
            return {
                "messages": [],
                "summary": None,
                "metadata": {"total_messages": 0, "returned_messages": 0},
                "status": "not_found"
            }
        
        # Extract messages
        messages = state.values.get("messages", [])
        
        # If messages list is empty, return not_found instead of error
        if not messages:
            logging.info(f"Empty conversation found for thread_id={thread_id}")
            return {
                "messages": [],
                "summary": state.values.get("summary"),
                "metadata": {
                    "total_messages": 0,
                    "returned_messages": 0,
                    "has_summary": state.values.get("summary") is not None
                },
                "status": "not_found"
            }
        
        # Format messages for output
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "type"):
                role = msg.type  # "human", "ai", "system"
            else:
                role = "unknown"
                
            formatted_messages.append({
                "role": role,
                "content": msg.content if hasattr(msg, "content") else str(msg),
                "id": msg.id if hasattr(msg, "id") else None
            })
        
        # Apply limit if specified
        if limit and len(formatted_messages) > limit:
            formatted_messages = formatted_messages[-limit:]
        
        # Extract metadata
        metadata = {
            "total_messages": len(messages),
            "returned_messages": len(formatted_messages),
            "has_summary": state.values.get("summary") is not None,
            "last_summarized_index": state.values.get("last_summarized_index"),
            "router_type": state.values.get("router", {}).get("type") if "router" in state.values else None
        }
        
        logging.info(f"Retrieved {len(formatted_messages)} messages for thread_id={thread_id}")
        
        return {
            "messages": formatted_messages,
            "summary": state.values.get("summary"),
            "metadata": metadata,
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error retrieving conversation history for thread_id={thread_id}: {e}", exc_info=True)
        return {
            "messages": [],
            "summary": None,
            "metadata": {"error": str(e)},
            "status": "error"
        }


async def delete_conversation_history(
    user_id: str,
    thread_id: str
) -> Dict[str, Any]:
    """
    Delete conversation history for a specific thread.
    
    Args:
        user_id: Unique identifier for the user
        thread_id: Unique identifier for the conversation thread
        
    Returns:
        Dictionary containing:
            - deleted: Boolean indicating success
            - message: Status message
            - status: Operation status
            
    Example:
        >>> result = await delete_conversation_history("user123", "thread456")
        >>> if result["deleted"]:
        ...     print("Conversation deleted successfully")
    """
    try:
        logging.info(f"Deleting conversation history for user_id={user_id}, thread_id={thread_id}")
        
        graph = get_graph()
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Check if conversation exists
        state = await graph.aget_state(config)
        
        if not state or not state.values:
            logging.info(f"No conversation found to delete for thread_id={thread_id}")
            return {
                "deleted": False,
                "message": "No conversation history found for this thread",
                "status": "not_found"
            }
        
        # Get message count before deletion
        message_count = len(state.values.get("messages", []))
        
        # Delete the conversation by updating state to empty
        await graph.aupdate_state(
            config,
            {
                "messages": [],
                "summary": None,
                "last_summarized_index": None,
                "router": None,
                "steps": [],
                "documents": []
            }
        )
        
        logging.info(f"Deleted {message_count} messages from thread_id={thread_id}")
        
        return {
            "deleted": True,
            "message": f"Successfully deleted {message_count} messages from conversation",
            "status": "success"
        }
        
    except Exception as e:
        logging.error(f"Error deleting conversation history for thread_id={thread_id}: {e}", exc_info=True)
        return {
            "deleted": False,
            "message": f"Error deleting conversation: {str(e)}",
            "status": "error"
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_rag_service():
        """Test the RAG service functions."""
        
        # Test parameters
        user_id = "test_user_001"
        thread_id = "test_thread_001"
        query = "How do I build a simple CNN model in TensorFlow?"
        
        print("=" * 70)
        print("Testing RAG Service")
        print("=" * 70)
        
        # Test 1: Execute RAG (non-streaming)
        print("\n1. Testing execute_rag...")
        result = await execute_rag(query, user_id, thread_id)
        print(f"Status: {result['status']}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Metadata: {result['metadata']}")
        
        # Test 2: Get conversation history
        print("\n2. Testing get_conversation_history...")
        history = await get_conversation_history(user_id, thread_id)
        print(f"Status: {history['status']}")
        print(f"Messages: {history['metadata']['total_messages']}")
        
        # Test 3: Execute RAG with streaming
        print("\n3. Testing execute_rag_stream...")
        async for chunk in execute_rag_stream(
            "What are the benefits of using tf.keras?",
            user_id,
            thread_id
        ):
            if chunk["type"] == "response_chunk":
                print(".", end="", flush=True)
            elif chunk["type"] == "node":
                print(f"\n[{chunk['node']}]", end="", flush=True)
            elif chunk["type"] == "end":
                print("\n✓ Completed")
        
        # Test 4: Get updated history
        print("\n4. Getting updated conversation history...")
        history = await get_conversation_history(user_id, thread_id, limit=5)
        print(f"Latest {len(history['messages'])} messages retrieved")
        
        # Test 5: Delete conversation
        print("\n5. Testing delete_conversation_history...")
        delete_result = await delete_conversation_history(user_id, thread_id)
        print(f"Deleted: {delete_result['deleted']}")
        print(f"Message: {delete_result['message']}")
        
        # Test 6: Verify deletion
        print("\n6. Verifying deletion...")
        history = await get_conversation_history(user_id, thread_id)
        print(f"Status: {history['status']}")
        print(f"Messages remaining: {len(history['messages'])}")
        
        print("\n" + "=" * 70)
        print("All tests completed!")
        print("=" * 70)
    
    # Run tests
    asyncio.run(test_rag_service())