from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status, Depends
from fastapi.responses import JSONResponse
from src.service import save_file_from_data_url
from src.service import index_pdf_file
from src.model import VercelChatRequest
from src.service import answer_question
from src.utils.http_headers import patch_vercel_headers
from src.utils.message_transformer import (
    extract_user_message,
    extract_file,
    check_file_media_type,
    check_messages_has_file,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.service import stream_any_langgraph_graph, stream_travel_system_chat
from src.core.agent.graph import get_qa_graph
from src.auth.jwt import verify_clerk_token
from src.core.retrieval.vector_store import _check_namespace_exists

app = FastAPI(
    title="Multi agent knowledge management system",
    description="A system to manage knowledge using multiple agents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Exception handler for unhandled exceptions.

    Args:
        request (Request): The incoming request.
        exc (Exception): The exception that occurred.

    Returns:
        JSONResponse: A JSON response with the error details.
    """

    if isinstance(exc, HTTPException):
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: VercelChatRequest, token=Depends(verify_clerk_token)):
    """
    Endpoint for processing user queries using the Vercel AI SDK format.

    Args:
        payload (VercelChatRequest): The request payload containing user messages and thread ID.
        token (_type_, optional): The authentication token. Defaults to Depends(verify_clerk_token).

    Raises:
        HTTPException: If user ID is not found in token.
        HTTPException: If no documents found. Please upload a document before asking questions.
        HTTPException: If file is not a PDF.

    Returns:
        StreamingResponse: A streaming response containing the AI's response to the user query.
    """
    # Extract user ID from token
    user_id = token.get("sub")  # Clerk uses 'sub' for user ID

    # Check if user ID is present in token
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token",
        )

    # Check if any message has a file part
    if check_messages_has_file(payload.messages):
        # Check if file is a PDF
        if not check_file_media_type(payload.messages):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a PDF",
            )

        filename, file_url = extract_file(payload.messages)

        if file_url:
            try:
                # Download and save the file with the original filename
                file_path = await save_file_from_data_url(file_url, filename)
                # Index the downloaded PDF into user's namespace
                await index_pdf_file(file_path, user_id)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process file: {str(e)}",
                )
    else:
        # Check if user has any documents in their namespace
        if not _check_namespace_exists(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents found. Please upload a document before asking questions.",
            )

    # Extract user message
    message = extract_user_message(payload.messages)

    # Delegate to the service layer which runs the multi-agent QA graph
    thread_id = payload.thread_id or payload.id

    response = StreamingResponse(
        stream_any_langgraph_graph(
            graph=get_qa_graph(),
            message=message,
            thread_id=thread_id,
            user_id=user_id,  # Pass user_id for namespace isolation
        ),
        media_type="text/event-stream",
    )

    return patch_vercel_headers(response)
