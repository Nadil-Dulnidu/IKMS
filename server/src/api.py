from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status, Depends
from fastapi.responses import JSONResponse
from src.service import save_uploaded_file, download_and_save_file
from src.service import index_pdf_file
from src.model import QuestionRequest, QAResponse, VercelChatRequest
from src.service import answer_question
from src.utils.http_headers import patch_vercel_headers
from src.utils.message_transformer import extract_user_message, extract_file
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.service import stream_any_langgraph_graph, stream_travel_system_chat
from src.core.agent.graph import get_qa_graph
from src.auth.jwt import verify_clerk_token

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

    if isinstance(exc, HTTPException):
        raise exc

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.post("/qa", status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: VercelChatRequest, token=Depends(verify_clerk_token)):
    # Extract user ID from token
    user_id = token.get("sub")  # Clerk uses 'sub' for user ID

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token",
        )

    # Extract user message
    message = extract_user_message(payload.messages)

    # Extract file info (filename and URL) if present
    filename, file_url = extract_file(payload.messages)

    # If file URL is present, download and index it to user's namespace
    if file_url:
        try:
            # Download and save the file with the original filename
            file_path = await download_and_save_file(file_url, filename)
            # Index the downloaded PDF into user's namespace
            await index_pdf_file(file_path, user_id)
        except Exception as e:
            print(f"Error processing file: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file: {str(e)}",
            )

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
