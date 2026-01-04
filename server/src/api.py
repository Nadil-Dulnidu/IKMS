from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from src.service import save_uploaded_file
from src.service import index_pdf_file
from src.model import QuestionRequest, QAResponse, VercelChatRequest
from src.service import answer_question
from src.utils.http_headers import patch_vercel_headers
from src.utils.message_transformer import extract_user_message
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.service import stream_any_langgraph_graph, stream_travel_system_chat
from src.core.agent.graph import get_qa_graph

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


@app.post("/index-pdf", status_code=status.HTTP_200_OK)
async def index_pdf(file: UploadFile = File(...)) -> dict | None:

    if file.content_type not in ("application/pdf",):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    file_path = await save_uploaded_file(file)
    await index_pdf_file(file_path)

    if file_path:
        return {
            "filename": file.filename,
            "message": "File uploaded successfully.",
        }


@app.post("/qa", status_code=status.HTTP_200_OK)
async def qa_endpoint(payload: VercelChatRequest):

    message = extract_user_message(payload.messages)

    # Delegate to the service layer which runs the multi-agent QA graph
    thread_id = payload.thread_id or payload.id
    print(f"Thread ID: {thread_id}")

    response = StreamingResponse(
        stream_any_langgraph_graph(
            graph=get_qa_graph(),
            message=message,
            thread_id=thread_id,
        ),
        media_type="text/event-stream",
    )

    return patch_vercel_headers(response)
