"""
domain/schemas.py
Data models for ROI analysis and proposal generation
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class LeafNodeAnalysis(BaseModel):
    """
    Analysis of leaf nodes in an ROI tree
    
    Attributes:
        has_numerical_data: Whether all leaf nodes have numerical data
        missing_nodes: List of names of nodes without numerical data
        incomplete_nodes: List of nodes with issues (name, issue, suggestion)
        completion_percentage: Completion percentage of the tree (0-100)
    """
    has_numerical_data: bool
    missing_nodes: List[str]
    incomplete_nodes: List[Dict[str, str]]
    completion_percentage: float

class ProposalResult(BaseModel):
    """
    Result of proposal generation
    
    Attributes:
        total_investment: Total investment cost
        total_benefit: Total expected benefit
        roi_percentage: ROI percentage
        implementation_timeframe: Estimated timeframe for implementation
        key_recommendations: List of key recommendations
        priority_ranking: Optional list of prioritized proposals
    """
    total_investment: float
    total_benefit: float
    roi_percentage: float
    implementation_timeframe: str
    key_recommendations: List[str]
    priority_ranking: Optional[List[Dict[str, Any]]] = None

class NumericalValue(BaseModel):
    """
    Numerical value with unit and type
    
    Attributes:
        value: Numerical value
        unit: Unit of the value (e.g. "円", "時間", "人", "件")
        type: Type of the value (e.g. "amount", "time", "count", "percentage")
        description: Optional description of the value
    """
    value: float
    unit: str
    type: str
    description: Optional[str] = None

class UnitConversion(BaseModel):
    """
    Conversion between different units
    
    Attributes:
        from_value: Source numerical value
        to_value: Target numerical value
        conversion_factor: Conversion factor
        explanation: Explanation of the conversion
    """
    from_value: NumericalValue
    to_value: NumericalValue
    conversion_factor: float
    explanation: str

class SolutionRecommendation(BaseModel):
    """
    Solution recommendation for a leaf node
    
    Attributes:
        leaf_node_id: ID of the leaf node
        leaf_node_name: Name of the leaf node
        solution_name: Name of the recommended solution
        solution_description: Description of the recommended solution
        estimated_cost: Estimated cost of the solution
        estimated_benefit: Estimated benefit of the solution
        implementation_timeframe: Timeframe for implementing the solution
        priority: Priority of the solution (1 = highest)
    """
    leaf_node_id: str
    leaf_node_name: str
    solution_name: str
    solution_description: str
    estimated_cost: NumericalValue
    estimated_benefit: NumericalValue
    implementation_timeframe: str
    priority: int