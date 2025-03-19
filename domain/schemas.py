"""
State schemas for the simplified ROI agents
"""
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field


class LeafNodeAnalysis(BaseModel):
    """Analysis of leaf nodes in an ROI tree"""
    has_numerical_data: bool
    missing_nodes: List[str] = []
    incomplete_nodes: List[Dict[str, Any]] = []
    completion_percentage: float = Field(..., ge=0, le=100)


class ProposalResult(BaseModel):
    """Result of proposal generation"""
    total_investment: float
    total_benefit: float
    roi_percentage: float
    implementation_timeframe: str
    key_recommendations: List[str]