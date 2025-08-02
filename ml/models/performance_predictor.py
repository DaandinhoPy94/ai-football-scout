import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

class PlayerPerformanceModel(nn.Module):
    """
    Multi-task model voor speler performance prediction
    """
    
    def __init__(
        self, 
        movement_features: int = 128,
        historical_features: int = 64,
        physical_features: int = 32
    ):
        super().__init__()
        
        # Movement encoder (from video analysis)
        self.movement_encoder = nn.LSTM(
            input_size=movement_features,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Historical performance encoder
        self.historical_encoder = nn.Sequential(
            nn.Linear(historical_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Physical attributes encoder
        self.physical_encoder = nn.Sequential(
            nn.Linear(physical_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        # Fusion layer
        fusion_size = 256 + 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.performance_head = nn.Linear(256, 1)  # Overall rating
        self.goals_head = nn.Linear(256, 1)  # Expected goals
        self.assists_head = nn.Linear(256, 1)  # Expected assists
        self.injury_risk_head = nn.Linear(256, 3)  # Low, Medium, High
        self.market_value_head = nn.Linear(256, 1)  # Transfer value
        
    def forward(
        self, 
        movement_seq: torch.Tensor,
        historical: torch.Tensor,
        physical: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Process movement sequence
        lstm_out, (h_n, c_n) = self.movement_encoder(movement_seq)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        movement_features = attn_out.mean(dim=1)  # Global average pooling
        
        # Process other features
        hist_features = self.historical_encoder(historical)
        phys_features = self.physical_encoder(physical)
        
        # Concatenate all features
        combined = torch.cat([
            movement_features,
            hist_features,
            phys_features
        ], dim=1)
        
        # Fusion
        fused = self.fusion(combined)
        
        # Predictions
        return {
            'performance_rating': torch.sigmoid(self.performance_head(fused)) * 100,
            'expected_goals': torch.relu(self.goals_head(fused)),
            'expected_assists': torch.relu(self.assists_head(fused)),
            'injury_risk': self.injury_risk_head(fused),
            'market_value': torch.relu(self.market_value_head(fused)) * 1e6
        }

class TacticalAnalysisModel:
    """
    Analyze team tactics and player roles
    """
    
    def __init__(self):
        self.formation_classifier = self._build_formation_classifier()
        self.role_classifier = self._build_role_classifier()
        
    def analyze_formation(
        self, 
        player_positions: np.ndarray,
        timestamp: float
    ) -> Dict[str, any]:
        """
        Detect team formation
        """
        # Normalize positions
        normalized = self._normalize_positions(player_positions)
        
        # Classify formation
        formation = self.formation_classifier.predict(normalized.reshape(1, -1))[0]
        confidence = self.formation_classifier.predict_proba(normalized.reshape(1, -1)).max()
        
        # Detect pressing intensity
        pressing = self._calculate_pressing_intensity(player_positions)
        
        # Defensive line height
        defensive_line = self._calculate_defensive_line(player_positions)
        
        return {
            'formation': formation,
            'confidence': confidence,
            'pressing_intensity': pressing,
            'defensive_line_height': defensive_line,
            'timestamp': timestamp
        }
    
    def identify_player_roles(
        self,
        player_movements: Dict[int, np.ndarray],
        formation: str
    ) -> Dict[int, str]:
        """
        Identify player roles based on movement patterns
        """
        roles = {}
        
        for player_id, movements in player_movements.items():
            # Extract movement features
            features = self._extract_movement_features(movements)
            
            # Classify role
            role = self.role_classifier.predict(features.reshape(1, -1))[0]
            roles[player_id] = role
        
        return roles
    
    def _extract_movement_features(self, movements: np.ndarray) -> np.ndarray:
        """
        Extract features from movement data
        """
        features = []
        
        # Average position
        avg_pos = np.mean(movements, axis=0)
        features.extend(avg_pos)
        
        # Movement range
        movement_range = np.ptp(movements, axis=0)
        features.extend(movement_range)
        
        # Speed statistics
        velocities = np.diff(movements, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        features.extend([
            np.mean(speeds),
            np.max(speeds),
            np.std(speeds)
        ])
        
        # Directional preferences
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        hist, _ = np.histogram(directions, bins=8, range=(-np.pi, np.pi))
        features.extend(hist / len(directions))
        
        return np.array(features)