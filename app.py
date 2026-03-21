from api_rag import app


if __name__ == "__main__":
    import uvicorn
    from api_rag.config import SETTINGS

    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.port)
