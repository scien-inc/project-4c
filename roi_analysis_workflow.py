"""
roi_analysis_workflow.py
Integrated workflow manager for ROI analysis
"""
from typing import Dict, List, Tuple, Optional, Any, Callable

from agents.deepdive_agent import ChallengeAgent
from agents.proposal_agent import ProposalAgent
from agents.unit_conversion_agent import UnitConversionAgent
from agents.node_prioritization_agent import NodePrioritizationAgent
from domain.roitree import ROINode, get_leaf_nodes, mermaid_to_roi_tree
from domain.schemas import (
    LeafNodeAnalysis,
    ProposalResult,
    PrioritizationResult,
    ConversionResult,
    MappingResult
)


class ROIAnalysisWorkflow:
    """Integrated workflow manager for ROI analysis"""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the ROI analysis workflow manager
        
        Args:
            model_name: Name of the OpenAI model to use for all agents
        """
        self.challenge_agent = ChallengeAgent(model_name)
        self.proposal_agent = ProposalAgent(model_name)
        self.unit_conversion_agent = UnitConversionAgent(model_name)
        self.node_prioritization_agent = NodePrioritizationAgent(model_name)
        self.model_name = model_name
    
    def analyze_challenge(self, challenge_text: str) -> Tuple[ROINode, str]:
        """
        Analyze a business challenge and create an ROI tree
        
        Args:
            challenge_text: Text describing the business challenge
            
        Returns:
            Tuple of (ROI tree root node, Mermaid diagram)
        """
        return self.challenge_agent.create_roi_tree(challenge_text)
    
    def collect_numerical_data(self, roi_tree: ROINode, mermaid_diagram: str) -> LeafNodeAnalysis:
        """
        Analyze numerical data in ROI tree leaf nodes
        
        Args:
            roi_tree: ROI tree root node
            mermaid_diagram: Mermaid diagram of the ROI tree
            
        Returns:
            LeafNodeAnalysis object with analysis results
        """
        return self.challenge_agent.analyze_leaf_nodes(roi_tree, mermaid_diagram)
    
    def prioritize_nodes(self, roi_tree: ROINode, user_priorities: str = "") -> PrioritizationResult:
        """
        Prioritize ROI tree leaf nodes based on ROI and user priorities
        
        Args:
            roi_tree: ROI tree root node
            user_priorities: Optional user priorities as text
            
        Returns:
            PrioritizationResult object with prioritized nodes and overall strategy
        """
        leaf_nodes = get_leaf_nodes(roi_tree)
        
        # Format leaf nodes for the prioritization agent
        formatted_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "value": node.value if node.value is not None else "不明"
            }
            for node in leaf_nodes
        ]
        
        # Prioritize nodes
        return self.node_prioritization_agent.prioritize_nodes(formatted_nodes, user_priorities)
    
    def analyze_for_unit_conversion(self, roi_tree: ROINode) -> Dict:
        """
        Analyze ROI tree for unit conversion needs
        
        Args:
            roi_tree: ROI tree root node
            
        Returns:
            Dictionary with analysis results
        """
        leaf_nodes = get_leaf_nodes(roi_tree)
        
        # Format leaf nodes for the unit conversion agent
        formatted_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "value": node.value if node.value is not None else "不明"
            }
            for node in leaf_nodes
        ]
        
        # Analyze for unit conversion
        return self.unit_conversion_agent.analyze_nodes(formatted_nodes)
    
    def convert_node_units(self, roi_tree: ROINode, collected_info: Dict) -> ConversionResult:
        """
        Convert node units based on collected information
        
        Args:
            roi_tree: ROI tree root node
            collected_info: Dictionary of collected additional information
            
        Returns:
            ConversionResult object with converted nodes and assumptions
        """
        leaf_nodes = get_leaf_nodes(roi_tree)
        
        # Format leaf nodes for the unit conversion agent
        formatted_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "value": node.value if node.value is not None else "不明"
            }
            for node in leaf_nodes
        ]
        
        # Convert node units
        return self.unit_conversion_agent.convert_nodes(formatted_nodes, collected_info)
    
    def create_2d_mapping(self, roi_tree: ROINode, x_axis: str, y_axis: str) -> MappingResult:
        """
        Map ROI tree nodes to a 2D coordinate system
        
        Args:
            roi_tree: ROI tree root node
            x_axis: Description of X axis
            y_axis: Description of Y axis
            
        Returns:
            MappingResult object with mapped nodes and quadrant analysis
        """
        leaf_nodes = get_leaf_nodes(roi_tree)
        
        # Format leaf nodes for the node prioritization agent
        formatted_nodes = [
            {
                "node_id": node.node_id,
                "name": node.name,
                "value": node.value if node.value is not None else "不明"
            }
            for node in leaf_nodes
        ]
        
        # Create 2D mapping
        return self.node_prioritization_agent.create_2d_mapping(formatted_nodes, x_axis, y_axis)
    
    def generate_proposal(self, roi_tree: ROINode, proposal_guidance: str = "", callback: Optional[Callable] = None) -> Tuple[str, ProposalResult]:
        """
        Generate proposal based on ROI tree
        
        Args:
            roi_tree: ROI tree root node
            proposal_guidance: Optional guidance for proposal generation
            callback: Optional streaming callback
            
        Returns:
            Tuple of (proposal text, ProposalResult object)
        """
        mermaid_diagram = roi_tree.get_full_mermaid()
        
        if callback:
            return self.proposal_agent.generate_proposal_streaming(mermaid_diagram, proposal_guidance, callback)
        else:
            return self.proposal_agent.generate_proposal(mermaid_diagram, proposal_guidance)
    
    def update_roi_tree(self, roi_tree: ROINode, conversion_result: ConversionResult) -> ROINode:
        """
        Update ROI tree with converted values
        
        Args:
            roi_tree: ROI tree root node
            conversion_result: ConversionResult object with converted nodes
            
        Returns:
            Updated ROI tree root node
        """
        for converted_node in conversion_result.converted_nodes:
            node = roi_tree.find_node_by_id(converted_node.node_id)
            if node:
                node.value = converted_node.converted_value
        
        return roi_tree