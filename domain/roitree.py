"""
Enhanced ROI tree implementation with importance factors and ROI calculation.
"""
from typing import Dict, List, Optional, Union, Any
import uuid


class ROINode:
    """
    ROI Tree Node with importance factors for weighted calculations.
    
    Attributes:
        name: Node name (e.g., "CostReduction", "RevenueIncrease")
        details: Additional information about the node
        importance_factor: Weight of this node relative to siblings (sum of siblings = 1.0)
        value: Estimated monetary value (if applicable)
        children: List of child nodes
        node_id: Unique identifier for the node
    """
    def __init__(
        self, 
        name: str, 
        details: Optional[str] = None,
        importance_factor: float = 1.0,
        value: Optional[float] = None
    ):
        self.name = name
        self.details = details
        self.importance_factor = importance_factor
        self.value = value
        self.children: List["ROINode"] = []
        self.node_id = str(uuid.uuid4())[:8]  # Short unique ID
    
    def add_child(self, child: "ROINode") -> "ROINode":
        """Add a child node and return it"""
        self.children.append(child)
        return child
    
    def find_node_by_id(self, node_id: str) -> Optional["ROINode"]:
        """Recursively find a node by its ID"""
        if self.node_id == node_id:
            return self
        
        for child in self.children:
            found = child.find_node_by_id(node_id)
            if found:
                return found
        
        return None
    
    def normalize_importance_factors(self) -> None:
        """
        Ensure that all siblings at each level have importance factors that sum to 1.0
        """
        if not self.children:
            return
            
        # Normalize child importance factors
        total = sum(child.importance_factor for child in self.children)
        if total > 0:  # Avoid division by zero
            for child in self.children:
                child.importance_factor = child.importance_factor / total
                
        # Recursively normalize grandchildren
        for child in self.children:
            child.normalize_importance_factors()
    
    def calculate_roi(self) -> Dict[str, Any]:
        """
        Calculate ROI for this node and its subtree
        
        Returns:
            Dictionary containing ROI calculations and breakdown
        """
        if not self.children:
            return {
                "node_id": self.node_id,
                "name": self.name,
                "value": self.value or 0,
                "weighted_value": self.value or 0,
                "children_values": []
            }
        
        children_calculations = []
        total_weighted_value = 0
        
        for child in self.children:
            child_calc = child.calculate_roi()
            weighted_value = child_calc["weighted_value"] * child.importance_factor
            total_weighted_value += weighted_value
            
            children_calculations.append({
                "node_id": child.node_id,
                "name": child.name,
                "raw_value": child_calc["weighted_value"],
                "importance_factor": child.importance_factor,
                "weighted_value": weighted_value,
                "children": child_calc.get("children_values", [])
            })
        
        return {
            "node_id": self.node_id,
            "name": self.name,
            "weighted_value": total_weighted_value,
            "children_values": children_calculations
        }
    
    def to_dict(self) -> Dict:
        """Convert the node to a dictionary for JSON serialization"""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "details": self.details,
            "importance_factor": self.importance_factor,
            "value": self.value,
            "children": [c.to_dict() for c in self.children]
        }
    
    def to_mermaid(self, parent_id: Optional[str] = None) -> List[str]:
        """
        Generate Mermaid diagram syntax for this node and its children
        
        Args:
            parent_id: Optional ID of parent node for connecting edges
            
        Returns:
            List of Mermaid diagram lines
        """
        node_id = f"node{self.node_id.replace('-', '')}"
        lines = [f"    {node_id}[\"{self.name}\"]"]
        
        # Add connection to parent
        if parent_id:
            lines.append(f"    {parent_id} --> {node_id}")
            
            # Add importance factor as edge label if not 1.0
            if self.importance_factor != 1.0 and self.importance_factor > 0:
                percentage = f"{self.importance_factor:.0%}"
                lines.append(f"    {parent_id} -- \"{percentage}\" --> {node_id}")
        
        # Process children recursively
        for child in self.children:
            child_lines = child.to_mermaid(node_id)
            lines.extend(child_lines)
            
        return lines
    
    def get_full_mermaid(self) -> str:
        """
        Generate complete Mermaid diagram representation
        
        Returns:
            Complete Mermaid diagram syntax for this tree
        """
        lines = ["graph TD"]
        lines.extend(self.to_mermaid())
        return "\n".join(lines)
    

def create_default_roi_tree() -> ROINode:
    """
    Create a default ROI tree with basic cost reduction and revenue increase nodes
    
    Returns:
        ROI tree with default structure
    """
    root = ROINode("Gain", "Business benefit")
    
    # Add default children with equal importance
    cost_reduction = root.add_child(
        ROINode("CostReduction", "Reducing operational expenses", importance_factor=0.5)
    )
    revenue_increase = root.add_child(
        ROINode("RevenueIncrease", "Growing top-line revenue", importance_factor=0.5)
    )
    
    return root