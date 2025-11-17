# ===================================================================================
# Project: ChatTensorflow
# File: src/routers/rag.py
# Description: FastAPI router for RAG endpoints
# Author: LALAN KUMAR
# Created: [16-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

import os
import sys
from typing import AsyncGenerator, Dict, Any
from fastapi import APIRouter, HTTPException, status, Body
from fastapi.responses import StreamingResponse

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.schemas.api.rag import (
    RAGQueryRequest,
    RAGResponse,
    ConversationHistoryRequest,
    ConversationHistoryResponse,
    DeleteConversationRequest,
    DeleteConversationResponse,
    ErrorResponse,
    StreamEvent,
    RAGMetadata,
    ConversationHistoryMetadata,
    ConversationMessage
)
from src.services.rag.rag_service import (
    execute_rag,
    execute_rag_stream,
    get_conversation_history,
    delete_conversation_history
)
from src.logger import logging

# Initialize router
router = APIRouter(
    prefix="/api/rag",
    tags=["RAG"],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)


# ========================== Helper Functions ==========================

def validate_request_fields(request_body: Dict[Any, Any]) -> tuple[str, str, str]:
    """
    Validate request and extract required fields.
    
    Args:
        request_body: Request body dictionary
    
    Returns:
        Tuple of (user_query, user_id, thread_id)
    
    Raises:
        HTTPException: If validation fails
    """
    # Handle both field name formats
    user_query = request_body.get("user_query") or request_body.get("message")
    user_id = request_body.get("user_id")
    thread_id = request_body.get("thread_id")
    
    if not user_query or not user_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_query/message is required and cannot be empty"
        )
    
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id is required and cannot be empty"
        )
    
    if not thread_id or not thread_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="thread_id is required and cannot be empty"
        )
    
    return user_query.strip(), user_id.strip(), thread_id.strip()


# ========================== Endpoints ==========================

@router.post(
    "/ask",
    response_model=RAGResponse,
    summary="Execute RAG query",
    description="Execute a RAG pipeline query and return the complete result"
)
async def ask(request: RAGQueryRequest) -> RAGResponse:
    """
    Execute RAG pipeline and return the final response.
    
    **Request Parameters:**
    - `user_query` (required): User's query
    - `user_id` (required): Unique identifier for the user
    - `thread_id` (required): Thread ID for conversation continuity
    
    **Response:**
    - `response`: Final response from the assistant
    - `metadata`: Additional information (sources, research steps, etc.)
    - `status`: Execution status
    """
    try:
        logging.info(
            f"Received RAG query request - "
            f"user_id: {request.user_id}, "
            f"thread_id: {request.thread_id}"
        )
        
        # Execute RAG pipeline
        result = await execute_rag(
            user_query=request.user_query,
            user_id=request.user_id,
            thread_id=request.thread_id
        )
        
        logging.info(f"RAG Result: {result}")
        
        # Check execution status
        if result["status"] == "error":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process query: {result['metadata'].get('error', 'Unknown error')}"
            )
        
        # Build response
        response = RAGResponse(
            response=result["response"],
            metadata=RAGMetadata(**result["metadata"]),
            status="success",
            timestamp=None  # Will be auto-generated
        )
        
        logging.info(
            f"Successfully processed RAG query - "
            f"thread_id: {request.thread_id}, "
            f"documents: {result['metadata']['documents_count']}"
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Unexpected error in ask: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )


@router.post(
    "/ask/stream",
    summary="Execute RAG query with streaming",
    description="Execute RAG pipeline with streaming output for real-time updates"
)
async def ask_stream(request_body: Dict[Any, Any] = Body(...)):
    """
    Execute RAG pipeline with streaming output.
    
    **Request Parameters:**
    - `user_query` or `message` (required): User's query
    - `user_id` (required): Unique identifier for the user
    - `thread_id` (required): Thread ID for conversation continuity
    
    **Response:**
    Returns Server-Sent Events (SSE) stream with:
    1. Node execution events with data updates
    2. Stream termination signal
    """
    try:
        # Validate request and extract fields
        user_query, user_id, thread_id = validate_request_fields(request_body)
        
        logging.info(
            f"Received streaming RAG query request - "
            f"user_id: {user_id}, "
            f"thread_id: {thread_id}, "
            f"query: {user_query[:50]}..."
        )
        
        # Create async generator for streaming
        async def stream_generator() -> AsyncGenerator[str, None]:
            """Generate SSE events for the stream."""
            
            try:
                # Stream execution updates
                async for chunk in execute_rag_stream(
                    user_query=user_query,
                    user_id=user_id,
                    thread_id=thread_id,
                    stream_mode="updates"
                ):
                    # chunk is a dict with keys: type, data, node
                    chunk_type = chunk.get("type")
                    
                    if chunk_type == "node":
                        # Node execution event
                        node_name = chunk.get("node", "unknown")
                        node_data = chunk.get("data", {})
                        
                        stream_event = {
                            "type": "node",  # Changed from event_type to type
                            "data": node_data,
                            "node": node_name
                        }
                        yield f"data: {StreamEvent(**stream_event).model_dump_json()}\n\n"
                    
                    elif chunk_type == "response_chunk":
                        # Response content update
                        node_name = chunk.get("node", "respond")
                        content = chunk.get("data", "")
                        
                        stream_event = {
                            "type": "response_chunk",  # Changed from event_type to type
                            "data": {
                                "messages": [{
                                    "content": content,
                                    "type": "ai"
                                }]
                            },
                            "node": node_name,
                        }
                        yield f"data: {StreamEvent(**stream_event).model_dump_json()}\n\n"
                    
                    elif chunk_type == "end":
                        # Completion event
                        stream_event = {
                            "type": "end",  # Changed from event_type to type
                            "data": chunk.get("data", {"status": "completed", "thread_id": thread_id}),
                            "node": "stream",
                        }
                        yield f"data: {StreamEvent(**stream_event).model_dump_json()}\n\n"
                    
                    elif chunk_type == "error":
                        # Error event
                        stream_event = {
                            "type": "error",  # Changed from event_type to type
                            "data": chunk.get("data", {"error": "Unknown error"}),
                            "node": "stream",
                        }
                        yield f"data: {StreamEvent(**stream_event).model_dump_json()}\n\n"
            
            except Exception as e:
                logging.error(f"Error in stream generator: {e}", exc_info=True)
                error_event = {
                    "type": "error",  # Changed from event_type to type
                    "data": {"error": str(e)},
                    "node": "stream",
                    "timestamp": None  # Will be auto-generated
                }
                yield f"data: {StreamEvent(**error_event).model_dump_json()}\n\n"
        
        # Return streaming response
        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable buffering for Nginx
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error initiating stream: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error initiating stream: {str(e)}"
        )


@router.post(
    "/history",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    description="Retrieve the conversation history for a specific thread"
)
async def get_history(request: ConversationHistoryRequest) -> ConversationHistoryResponse:
    """
    Get conversation history for a thread.
    
    **Request Parameters:**
    - `user_id` (required): User identifier
    - `thread_id` (required): Thread ID to retrieve
    - `limit` (optional): Maximum number of recent messages to return
    
    **Response:**
    - Full conversation history with messages and metadata
    """
    try:
        logging.info(
            f"Retrieving conversation history - "
            f"user_id: {request.user_id}, "
            f"thread_id: {request.thread_id}, "
            f"limit: {request.limit}"
        )
        
        # Get conversation history
        result = await get_conversation_history(
            user_id=request.user_id,
            thread_id=request.thread_id,
            limit=request.limit
        )
        
        # Convert messages to schema format
        messages = [
            ConversationMessage(**msg)
            for msg in result["messages"]
        ]
        
        # Build response
        response = ConversationHistoryResponse(
            messages=messages,
            summary=result["summary"],
            metadata=ConversationHistoryMetadata(**result["metadata"]),
            status=result["status"],
            timestamp=None  # Will be auto-generated
        )
        
        logging.info(
            f"Retrieved conversation history - "
            f"thread_id: {request.thread_id}, "
            f"messages: {len(messages)}, "
            f"status: {result['status']}"
        )
        
        return response
    
    except Exception as e:
        logging.error(f"Error retrieving history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving history: {str(e)}"
        )


@router.delete(
    "/history",
    response_model=DeleteConversationResponse,
    summary="Delete conversation history",
    description="Delete all messages and state for a specific conversation thread"
)
async def delete_history(request: DeleteConversationRequest) -> DeleteConversationResponse:
    """
    Delete conversation history for a thread.
    
    **Request Parameters:**
    - `user_id` (required): User identifier  
    - `thread_id` (required): Thread ID to delete
    
    **Response:**
    - Deletion status and message
    """
    try:
        logging.info(
            f"Deleting conversation history - "
            f"user_id: {request.user_id}, "
            f"thread_id: {request.thread_id}"
        )
        
        # Delete conversation
        result = await delete_conversation_history(
            user_id=request.user_id,
            thread_id=request.thread_id
        )
        
        # Build response
        response = DeleteConversationResponse(
            deleted=result["deleted"],
            message=result["message"],
            status=result["status"],
            timestamp=None  # Will be auto-generated
        )
        
        logging.info(
            f"Conversation deletion result - "
            f"thread_id: {request.thread_id}, "
            f"deleted: {result['deleted']}, "
            f"status: {result['status']}"
        )
        
        return response
    
    except Exception as e:
        logging.error(f"Error deleting history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting history: {str(e)}"
        )


@router.get(
    "/health",
    summary="Health check",
    description="Check the health status of the RAG service"
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dictionary with health status information
    """
    try:
        return {
            "status": "healthy",
            "service": "tensorflow-rag",
            "version": "1.0.0"
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )

