from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
from scanner_engine import scanner

app = FastAPI(title="QR Code Scanner API")

@app.post("/scan-qr")
async def scan_qr(file: UploadFile = File(...)):
    """
    Scans an uploaded image and returns the raw QR code value.
    """
    try:
        content = await file.read()
        result = scanner.scan(content)
        
        if result["status"] == "failed":
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)