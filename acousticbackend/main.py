import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from acoustic_extractor import extract_acoustic_readings

app = FastAPI(title="EchoHealth Dashboard Backend")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_files"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/extract-features/")
async def extract_features(file: UploadFile = File(...)):
    try:
        # save upload
        tmp = os.path.join(TEMP_DIR, f"tmp_{file.filename}")
        with open(tmp, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        feats = extract_acoustic_readings(tmp)
        os.remove(tmp)
        return {"success": True, "data": feats}

    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
