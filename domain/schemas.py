"""
State schemas for the ROI agents
"""
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, FunctionMessage
from pydantic import BaseModel, Field

# Type definition for node path (to track position in tree)
NodePath = List[str]

# =========================
# Pydantic models for better type safety
# =========================

class ROINodeUpdate(BaseModel):
    """Represents a parsed update to the ROI tree"""
    parent_node_id: Optional[str] = None
    name: str
    details: Optional[str] = None
    importance_factor: float = 1.0
    value: Optional[float] = None
    

class ROIAnalysis(BaseModel):
    """Self-reflection analysis of the ROI tree exploration progress"""
    deepdive_needed: bool
    reason: str
    suggested_focus: Optional[str] = None
    deepdive_completion_percentage: float = Field(..., ge=0, le=100)
    

class ROICalculation(BaseModel):
    """ROI calculation for a proposal"""
    node_id: str
    name: str  
    estimated_value: float
    confidence: float = Field(..., ge=0, le=1)
    assumptions: List[str] = []
    

class ProposalAnalysis(BaseModel):
    """Self-reflection analysis of the proposal generation process"""
    proposal_complete: bool
    reason: str
    missing_information: List[str] = []
    roi_confidence: float = Field(..., ge=0, le=1)


# =========================
# State type definitions
# =========================

class DeepdiveState(TypedDict):
    # Messages and conversation history
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, FunctionMessage]]
    
    # ROI tree data
    root_node: Any  # ROINode object (typed as Any to avoid circular imports)
    current_node_id: Optional[str]  # ID of the node being explored
    
    # Path tracking for navigation
    node_path: NodePath
    
    # Analysis and control flags
    exploration_history: List[str]  # IDs of nodes already explored
    iteration_count: int
    max_iterations: int
    min_nodes_per_branch: int  # Minimum depth to explore
    self_reflection: Optional[ROIAnalysis]  # Last self-reflection result
    exploration_complete: bool


class ProposalState(TypedDict):
    # Messages and conversation history
    messages: List[Union[HumanMessage, AIMessage, SystemMessage, FunctionMessage]]
    
    # Reference to the ROI tree (readonly)
    root_node: Any  # ROINode object
    
    # ROI calculation results
    roi_calculations: Dict[str, Any]
    current_node_id: Optional[str]  # ID of the node being analyzed
    
    # Analysis and control flags
    analyzed_nodes: List[str]  # IDs of nodes already analyzed
    iteration_count: int
    max_iterations: int
    self_reflection: Optional[ProposalAnalysis]
    proposal_complete: bool