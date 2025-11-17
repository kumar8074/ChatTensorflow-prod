
# ===================================================================================
# Project: ChatTensorflow
# File: src/schemas/api/rag.py
# Description: Pydantic schemas for RAG API endpoints
# Author: LALAN KUMAR
# Created: [16-11-2025]
# Updated: [16-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query execution."""
    
    user_query: str = Field(
        ...,
        description="The user's question or query about TensorFlow",
        min_length=1,
        max_length=2000,
        examples=["How do I build a CNN model in TensorFlow?"]
    )
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
        max_length=100,
        examples=["user_12345"]
    )
    thread_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        min_length=1,
        max_length=100,
        examples=["thread_abc123"]
    )
    
    @field_validator("user_query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean the user query."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or only whitespace")
        return v
    
    @field_validator("user_id", "thread_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ID formats."""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty or only whitespace")
        # Remove any potentially problematic characters
        if any(char in v for char in [" ", "\n", "\t", "/", "\\"]):
            raise ValueError("ID cannot contain spaces or special path characters")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_query": "How do I create a sequential model in TensorFlow?",
                    "user_id": "user_001",
                    "thread_id": "thread_001"
                }
            ]
        }
    }


class ConversationHistoryRequest(BaseModel):
    """Request schema for retrieving conversation history."""
    
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
        max_length=100
    )
    thread_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        min_length=1,
        max_length=100
    )
    limit: Optional[int] = Field(
        None,
        description="Maximum number of recent messages to return",
        ge=1,
        le=1000,
        examples=[10, 50, 100]
    )
    
    @field_validator("user_id", "thread_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ID formats."""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        if any(char in v for char in [" ", "\n", "\t", "/", "\\"]):
            raise ValueError("ID cannot contain spaces or special path characters")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "thread_id": "thread_001",
                    "limit": 20
                }
            ]
        }
    }


class DeleteConversationRequest(BaseModel):
    """Request schema for deleting conversation history."""
    
    user_id: str = Field(
        ...,
        description="Unique identifier for the user",
        min_length=1,
        max_length=100
    )
    thread_id: str = Field(
        ...,
        description="Unique identifier for the conversation thread",
        min_length=1,
        max_length=100
    )
    
    @field_validator("user_id", "thread_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate ID formats."""
        v = v.strip()
        if not v:
            raise ValueError("ID cannot be empty")
        if any(char in v for char in [" ", "\n", "\t", "/", "\\"]):
            raise ValueError("ID cannot contain spaces or special path characters")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_001",
                    "thread_id": "thread_001"
                }
            ]
        }
    }



class DocumentMetadata(BaseModel):
    """Metadata for a retrieved document."""
    
    source_url: str = Field(description="URL of the source document")
    page_type: Optional[str] = Field(None, description="Type of page (tutorial, api, etc.)")
    heading: Optional[str] = Field(None, description="Section heading")
    title: Optional[str] = Field(None, description="Page title")
    breadcrumbs: List[str] = Field(default_factory=list, description="Navigation breadcrumbs")
    has_code: bool = Field(False, description="Whether document contains code")


class RAGMetadata(BaseModel):
    """Metadata about the RAG execution."""
    
    router_type: Optional[Literal["tensorflow", "more-info", "general"]] = Field(
        None,
        description="Type of query routing"
    )
    router_logic: Optional[str] = Field(
        None,
        description="Logic used for routing decision"
    )
    research_steps: List[str] = Field(
        default_factory=list,
        description="Research steps taken"
    )
    documents_count: int = Field(
        0,
        description="Number of documents retrieved"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of source URLs used"
    )
    summary: Optional[str] = Field(
        None,
        description="Conversation summary if available"
    )
    message_count: int = Field(
        0,
        description="Total number of messages in conversation"
    )


class RAGResponse(BaseModel):
    """Response schema for RAG query execution."""
    
    response: str = Field(
        ...,
        description="The AI-generated response to the user's query"
    )
    metadata: RAGMetadata = Field(
        ...,
        description="Metadata about the RAG execution"
    )
    status: Literal["success", "error"] = Field(
        ...,
        description="Status of the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the response"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "To create a CNN model in TensorFlow, you can use...",
                    "metadata": {
                        "router_type": "tensorflow",
                        "router_logic": "Query is about TensorFlow CNN implementation",
                        "research_steps": ["Search CNN documentation", "Find examples"],
                        "documents_count": 5,
                        "sources": ["https://tensorflow.org/..."],
                        "summary": None,
                        "message_count": 2
                    },
                    "status": "success",
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }


class StreamEvent(BaseModel):
    """Schema for streaming events."""
    
    type: Literal["start", "node", "response_chunk", "end", "error"] = Field(
        ...,
        description="Type of streaming event"
    )
    data: Any = Field(
        ...,
        description="Event-specific data"
    )
    node: Optional[str] = Field(
        None,
        description="Current node being executed (if applicable)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the event"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "response_chunk",
                    "data": "To build a CNN model...",
                    "node": "respond",
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }


class ConversationMessage(BaseModel):
    """Schema for a single conversation message."""
    
    role: Literal["human", "ai", "system", "unknown"] = Field(
        ...,
        description="Role of the message sender"
    )
    content: str = Field(
        ...,
        description="Content of the message"
    )
    id: Optional[str] = Field(
        None,
        description="Unique message ID"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "role": "human",
                    "content": "How do I build a CNN?",
                    "id": "msg_123"
                }
            ]
        }
    }


class ConversationHistoryMetadata(BaseModel):
    """Metadata for conversation history."""
    
    total_messages: int = Field(
        ...,
        description="Total number of messages in the thread"
    )
    returned_messages: int = Field(
        ...,
        description="Number of messages returned in this response"
    )
    has_summary: bool = Field(
        False,
        description="Whether a conversation summary exists"
    )
    last_summarized_index: Optional[int] = Field(
        None,
        description="Index of last summarized message"
    )
    router_type: Optional[str] = Field(
        None,
        description="Last router type used"
    )


class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history retrieval."""
    
    messages: List[ConversationMessage] = Field(
        default_factory=list,
        description="List of conversation messages"
    )
    summary: Optional[str] = Field(
        None,
        description="Conversation summary if available"
    )
    metadata: ConversationHistoryMetadata = Field(
        ...,
        description="Metadata about the conversation"
    )
    status: Literal["success", "not_found", "error"] = Field(
        ...,
        description="Status of the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the response"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "human",
                            "content": "How do I build a CNN?",
                            "id": "msg_1"
                        },
                        {
                            "role": "ai",
                            "content": "To build a CNN in TensorFlow...",
                            "id": "msg_2"
                        }
                    ],
                    "summary": "Discussion about building CNN models",
                    "metadata": {
                        "total_messages": 2,
                        "returned_messages": 2,
                        "has_summary": True,
                        "last_summarized_index": 0,
                        "router_type": "tensorflow"
                    },
                    "status": "success",
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }


class DeleteConversationResponse(BaseModel):
    """Response schema for conversation deletion."""
    
    deleted: bool = Field(
        ...,
        description="Whether the conversation was successfully deleted"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    status: Literal["success", "not_found", "error"] = Field(
        ...,
        description="Status of the request"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the response"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "deleted": True,
                    "message": "Successfully deleted 5 messages from conversation",
                    "status": "success",
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    field: Optional[str] = Field(
        None,
        description="Field that caused the error (if applicable)"
    )
    message: str = Field(
        ...,
        description="Error message"
    )
    type: str = Field(
        ...,
        description="Error type"
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    status: Literal["error"] = Field(
        "error",
        description="Status indicating an error occurred"
    )
    message: str = Field(
        ...,
        description="Human-readable error message"
    )
    details: Optional[List[ErrorDetail]] = Field(
        None,
        description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the error"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "error",
                    "message": "Invalid request parameters",
                    "details": [
                        {
                            "field": "user_query",
                            "message": "Query cannot be empty",
                            "type": "value_error"
                        }
                    ],
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    
    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Health status of the service"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    services: Dict[str, Literal["up", "down"]] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the health check"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "services": {
                        "opensearch": "up",
                        "llm": "up",
                        "embedding": "up"
                    },
                    "timestamp": "2025-11-16T10:30:00Z"
                }
            ]
        }
    }