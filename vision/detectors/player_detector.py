import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Tuple, Optional
import torch
from dataclasses import dataclass
import mediapipe as mp

@dataclass
class Player:
    """Player detection result"""
    id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    team: Optional[str] = None
    jersey_number: Optional[int] = None
    pose_keypoints: Optional[np.ndarray] = None

class FootballVisionPipeline:
    """
    Complete vision pipeline for football analysis
    """
    
    def __init__(self, model_path: str = "yolov8x.pt"):
        # Initialize YOLO for object detection
        self.yolo = YOLO(model_path)
        
        # Initialize MediaPipe for pose estimation
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
        
        # Team classifier (trained separately)
        self.team_classifier = self._load_team_classifier()
        
        # Jersey number OCR
        self.jersey_ocr = self._load_jersey_ocr()
        
    def process_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Process single frame
        """
        # Detect objects
        detections = self._detect_objects(frame)
        
        # Track players
        tracked_players = self._track_players(detections, frame)
        
        # Estimate poses
        players_with_poses = self._estimate_poses(tracked_players, frame)
        
        # Classify teams
        players_with_teams = self._classify_teams(players_with_poses, frame)
        
        # Detect ball
        ball_position = self._detect_ball(detections)
        
        # Detect events
        events = self._detect_events(players_with_teams, ball_position)
        
        return {
            "players": players_with_teams,
            "ball": ball_position,
            "events": events,
            "frame_number": self.frame_count
        }
    
    def _detect_objects(self, frame: np.ndarray) -> sv.Detections:
        """
        Detect players, ball, and other objects
        """
        results = self.yolo(frame, conf=0.3)
        
        # Filter for relevant classes (person, sports ball)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # Filter by class
        person_mask = detections.class_id == 0  # person class
        ball_mask = detections.class_id == 32  # sports ball class
        
        return {
            "persons": detections[person_mask],
            "balls": detections[ball_mask]
        }
    
    def _track_players(self, detections: Dict, frame: np.ndarray) -> List[Player]:
        """
        Track players across frames
        """
        # Update tracker
        tracked = self.tracker.update_with_detections(detections["persons"])
        
        players = []
        for i, (bbox, track_id, conf) in enumerate(zip(
            tracked.xyxy, 
            tracked.tracker_id, 
            tracked.confidence
        )):
            player = Player(
                id=int(track_id),
                bbox=tuple(bbox.astype(int)),
                confidence=float(conf)
            )
            players.append(player)
        
        return players
    
    def _estimate_poses(self, players: List[Player], frame: np.ndarray) -> List[Player]:
        """
        Estimate pose for each player
        """
        for player in players:
            x1, y1, x2, y2 = player.bbox
            
            # Crop player region
            player_crop = frame[y1:y2, x1:x2]
            
            # Convert to RGB for MediaPipe
            player_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(player_rgb)
            
            if results.pose_landmarks:
                # Convert to numpy array
                keypoints = np.array([
                    [lm.x * (x2-x1) + x1, 
                     lm.y * (y2-y1) + y1, 
                     lm.z, 
                     lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ])
                player.pose_keypoints = keypoints
        
        return players
    
    def _classify_teams(self, players: List[Player], frame: np.ndarray) -> List[Player]:
        """
        Classify players into teams based on jersey colors
        """
        for player in players:
            x1, y1, x2, y2 = player.bbox
            
            # Get jersey region (upper body)
            jersey_y1 = y1 + int((y2-y1) * 0.2)
            jersey_y2 = y1 + int((y2-y1) * 0.6)
            jersey_region = frame[jersey_y1:jersey_y2, x1:x2]
            
            # Classify team
            if jersey_region.size > 0:
                team = self.team_classifier.predict(jersey_region)
                player.team = team
                
                # Try to detect jersey number
                number = self.jersey_ocr.detect(jersey_region)
                if number:
                    player.jersey_number = number
        
        return players
    
    def _detect_events(self, players: List[Player], ball_pos: Optional[Tuple]) -> List[Dict]:
        """
        Detect game events (passes, shots, tackles, etc.)
        """
        events = []
        
        if ball_pos:
            # Find player closest to ball
            min_dist = float('inf')
            ball_player = None
            
            for player in players:
                player_center = self._get_bbox_center(player.bbox)
                dist = np.linalg.norm(
                    np.array(player_center) - np.array(ball_pos)
                )
                
                if dist < min_dist:
                    min_dist = dist
                    ball_player = player
            
            # Detect events based on pose and ball proximity
            if ball_player and ball_player.pose_keypoints is not None:
                event = self._analyze_player_action(
                    ball_player, 
                    ball_pos, 
                    players
                )
                if event:
                    events.append(event)
        
        return events
    
    def _analyze_player_action(
        self, 
        player: Player, 
        ball_pos: Tuple, 
        all_players: List[Player]
    ) -> Optional[Dict]:
        """
        Analyze player action based on pose and context
        """
        # Get relevant keypoints
        keypoints = player.pose_keypoints
        
        # Foot positions
        left_foot = keypoints[31][:2]  # MediaPipe left foot
        right_foot = keypoints[32][:2]  # MediaPipe right foot
        
        # Check if kicking
        foot_to_ball_dist = min(
            np.linalg.norm(left_foot - ball_pos),
            np.linalg.norm(right_foot - ball_pos)
        )
        
        if foot_to_ball_dist < 50:  # pixels
            # Determine action type
            action_type = self._classify_action(player, all_players)
            
            return {
                "type": action_type,
                "player_id": player.id,
                "team": player.team,
                "position": ball_pos,
                "confidence": 0.8
            }
        
        return None

# Advanced video processor
class VideoProcessor:
    """
    Process full match videos
    """
    
    def __init__(self, vision_pipeline: FootballVisionPipeline):
        self.vision = vision_pipeline
        self.results = []
        
    async def process_video(self, video_path: str, output_path: str = None):
        """
        Process entire video file
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output requested
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame_results = self.vision.process_frame(frame)
                frame_results['timestamp'] = frame_count / fps
                self.results.append(frame_results)
                
                # Draw annotations
                annotated_frame = self._draw_annotations(frame, frame_results)
                
                # Write output
                if output_path:
                    out.write(annotated_frame)
                
                frame_count += 1
                pbar.update(1)
                
                # Process in batches for memory efficiency
                if frame_count % 1000 == 0:
                    await self._save_batch_results()
                    self.results = []
        
        cap.release()
        if output_path:
            out.release()
        
        # Save final batch
        await self._save_batch_results()
        
        return {
            "total_frames": frame_count,
            "duration": frame_count / fps,
            "fps": fps
        }
    
    def _draw_annotations(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw bounding boxes, poses, and events
        """
        annotated = frame.copy()
        
        # Draw players
        for player in results["players"]:
            x1, y1, x2, y2 = player.bbox
            
            # Team color
            color = (0, 255, 0) if player.team == "home" else (0, 0, 255)
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Player ID and number
            label = f"#{player.id}"
            if player.jersey_number:
                label += f" ({player.jersey_number})"
            
            cv2.putText(
                annotated, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
            
            # Draw pose
            if player.pose_keypoints is not None:
                self._draw_pose(annotated, player.pose_keypoints)
        
        # Draw ball
        if results["ball"]:
            cv2.circle(
                annotated, 
                tuple(map(int, results["ball"])), 
                10, 
                (255, 255, 0), 
                -1
            )
        
        # Draw events
        for event in results["events"]:
            cv2.putText(
                annotated, 
                event["type"], 
                tuple(map(int, event["position"])),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2
            )
        
        return annotated
    