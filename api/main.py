# api/main.py
from fastapi import FastAPI, UploadFile, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import asyncio
from celery import Celery
import redis
from sqlalchemy.ext.asyncio import AsyncSession
import json

app = FastAPI(title="AI Football Scout API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for real-time updates
redis_client = redis.asyncio.Redis(host='localhost', port=6379, decode_responses=True)

# WebSocket manager for live updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        del self.active_connections[client_id]

    async def send_update(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

# Endpoints
@app.post("/api/v1/analyze/video")
async def analyze_video(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    match_data: Optional[Dict] = None
):
    """
    Upload and analyze match video
    """
    # Save video
    video_path = f"uploads/{file.filename}"
    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create analysis job
    job_id = str(uuid.uuid4())
    
    # Start async processing
    background_tasks.add_task(
        process_video_analysis,
        job_id,
        video_path,
        match_data
    )
    
    return {
        "job_id": job_id,
        "status": "processing",
        "websocket_url": f"/ws/{job_id}"
    }

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket for real-time analysis updates
    """
    await manager.connect(websocket, job_id)
    
    try:
        while True:
            # Send progress updates
            progress = await redis_client.get(f"progress:{job_id}")
            if progress:
                await manager.send_update(job_id, json.loads(progress))
            
            await asyncio.sleep(1)
            
    except Exception as e:
        manager.disconnect(job_id)

@app.get("/api/v1/players/{player_id}/analysis")
async def get_player_analysis(
    player_id: str,
    time_range: Optional[str] = "30d"
):
    """
    Get comprehensive player analysis
    """
    # Fetch from database
    analysis = await fetch_player_analysis(player_id, time_range)
    
    return {
        "player_id": player_id,
        "performance_metrics": analysis["metrics"],
        "video_insights": analysis["video_insights"],
        "comparison": analysis["peer_comparison"],
        "predictions": analysis["predictions"],
        "highlights": analysis["top_highlights"]
    }

@app.post("/api/v1/reports/generate")
async def generate_scouting_report(
    player_id: str,
    report_type: str = "comprehensive",
    include_video: bool = True
):
    """
    Generate AI-powered scouting report
    """
    # Get player data
    player_data = await fetch_player_data(player_id)
    video_analysis = await fetch_video_analysis(player_id)
    performance_metrics = await fetch_performance_metrics(player_id)
    
    # Generate report
    generator = ScoutingReportGenerator(
        openai_api_key=settings.openai_api_key,
        pinecone_api_key=settings.pinecone_api_key
    )
    
    report = await generator.generate_report(
        player_data,
        video_analysis,
        performance_metrics
    )
    
    return {
        "report_url": report["pdf_path"],
        "key_insights": report["report"]["key_insights"],
        "visualizations": report["visualizations"],
        "similar_players": report["report"]["comparisons"]
    }

@app.post("/api/v1/highlights/mint-nft")
async def mint_highlight_nft(
    highlight_id: str,
    recipient_address: str
):
    """
    Mint NFT for a special highlight
    """
    # Get highlight data
    highlight = await fetch_highlight(highlight_id)
    
    # Check if eligible for NFT
    if highlight["rarity_score"] < 8:
        raise HTTPException(
            status_code=400,
            detail="Highlight must have rarity score >= 8"
        )
    
    # Mint NFT
    minter = HighlightNFTMinter(
        web3_provider=settings.web3_provider,
        contract_address=settings.nft_contract,
        private_key=settings.minter_private_key
    )
    
    result = await minter.mint_highlight(
        highlight["video_path"],
        {
            "title": highlight["title"],
            "description": highlight["description"],
            "player_name": highlight["player_name"],
            "team": highlight["team"],
            "match": highlight["match"],
            "event_type": highlight["event_type"],
            "rarity": highlight["rarity_score"],
            "timestamp": highlight["timestamp"]
        },
        recipient_address
    )
    
    return result

# Background tasks
async def process_video_analysis(job_id: str, video_path: str, match_data: Dict):
    """
    Process video analysis pipeline
    """
    try:
        # Initialize pipeline
        vision_pipeline = FootballVisionPipeline()
        processor = VideoProcessor(vision_pipeline)
        
        # Process video
        await redis_client.set(
            f"progress:{job_id}",
            json.dumps({"status": "processing", "progress": 0})
        )
        
        results = await processor.process_video(video_path)
        
        # Extract insights
        insights = await extract_insights(results)
        
        # Store results
        await store_analysis_results(job_id, insights)
        
        # Update status
        await redis_client.set(
            f"progress:{job_id}",
            json.dumps({"status": "completed", "progress": 100})
        )
        
    except Exception as e:
        await redis_client.set(
            f"progress:{job_id}",
            json.dumps({"status": "failed", "error": str(e)})
        )