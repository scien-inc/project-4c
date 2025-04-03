"""
domain/roitree.py
Simplified ROI tree implementation for business analysis.
"""
from typing import Dict, List, Optional, Any
import uuid


class ROINode:
    """
    ROI Tree Node representing business challenges or proposals
    
    Attributes:
        name: Node name (e.g., "CostReduction", "RevenueIncrease")
        details: Additional information about the node
        value: Monetary value associated with the node (target or expected benefit)
        children: List of child nodes
        node_id: Unique identifier for the node
    """
    def __init__(
        self, 
        name: str, 
        details: Optional[str] = None,
        value: Optional[float] = None
    ):
        self.name = name
        self.details = details
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
                "children_values": []
            }
        
        children_calculations = []
        total_value = 0
        
        for child in self.children:
            child_calc = child.calculate_roi()
            child_value = child_calc["value"]
            total_value += child_value
            
            children_calculations.append({
                "node_id": child.node_id,
                "name": child.name,
                "value": child_value,
                "children": child_calc.get("children_values", [])
            })
        
        return {
            "node_id": self.node_id,
            "name": self.name,
            "value": total_value,
            "children_values": children_calculations
        }
    
    def to_dict(self) -> Dict:
        """Convert the node to a dictionary for JSON serialization"""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "details": self.details,
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
        
        # Add value to node label if available
        node_label = self.name
        if self.value is not None:
            node_label += f" ({format_value(self.value)})"
            
        lines = [f"    {node_id}[\"{node_label}\"]"]
        
        # Add connection to parent
        if parent_id:
            lines.append(f"    {parent_id} --> {node_id}")
        
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


def format_value(value: float) -> str:
    """Format monetary value for display"""
    if abs(value) >= 10000000:  # 1千万以上
        return f"{value/10000000:.1f}億円"
    elif abs(value) >= 10000:  # 1万以上
        return f"{value/10000:.1f}万円"
    else:
        return f"{value:,.0f}円"


def create_default_roi_tree() -> ROINode:
    """
    Create a default ROI tree with basic cost reduction and revenue increase nodes
    
    Returns:
        ROI tree with default structure
    """
    root = ROINode("ROI", "Business benefit")
    
    # Add default children
    cost_reduction = root.add_child(
        ROINode("コスト削減", "Reducing operational expenses")
    )
    revenue_increase = root.add_child(
        ROINode("売上拡大", "Growing top-line revenue")
    )
    
    return root


def mermaid_to_roi_tree(mermaid_text: str) -> Optional[ROINode]:
    """
    Parse a Mermaid diagram into an ROI tree
    
    Args:
        mermaid_text: Mermaid diagram text
        
    Returns:
        ROI tree root node or None if parsing fails
    """
    try:
        lines = [line.strip() for line in mermaid_text.split('\n') if line.strip()]
        
        # Skip the 'graph TD' line
        if lines[0].startswith('graph '):
            lines = lines[1:]
        
        # First, identify all nodes
        nodes = {}
        node_pattern = r'\s*(\w+)\[\"([^\"]+)\"\]'
        import re
        
        for line in lines:
            match = re.match(node_pattern, line)
            if match:
                node_id = match.group(1)
                node_label = match.group(2)
                
                # Extract value if present
                value = None
                value_match = re.search(r'\(([\d\.]+[億万]?円)\)', node_label)
                if value_match:
                    value_str = value_match.group(1)
                    node_label = node_label.replace(f" ({value_str})", "")
                    
                    # Convert to numeric value
                    if '億円' in value_str:
                        value = float(value_str.replace('億円', '')) * 100000000
                    elif '万円' in value_str:
                        value = float(value_str.replace('万円', '')) * 10000
                    else:
                        value = float(value_str.replace('円', '').replace(',', ''))
                
                nodes[node_id] = ROINode(node_label, value=value)
        
        # Then, establish parent-child relationships
        edge_pattern = r'\s*(\w+)\s*-->\s*(\w+)'
        
        for line in lines:
            match = re.match(edge_pattern, line)
            if match:
                parent_id = match.group(1)
                child_id = match.group(2)
                
                if parent_id in nodes and child_id in nodes:
                    nodes[parent_id].add_child(nodes[child_id])
        
        # Identify the root node (has no parents)
        parents = set()
        children = set()
        
        for line in lines:
            match = re.match(edge_pattern, line)
            if match:
                parents.add(match.group(1))
                children.add(match.group(2))
        
        root_candidates = parents - children
        if root_candidates:
            root_id = next(iter(root_candidates))
            return nodes[root_id]
        
        # If no clear root, just return the first node
        return next(iter(nodes.values()))
    
    except Exception as e:
        print(f"Error parsing Mermaid diagram: {str(e)}")
        return None


def get_leaf_nodes(node: ROINode) -> List[ROINode]:
    """Get all leaf nodes from the tree"""
    if not node.children:
        return [node]
    
    leaves = []
    for child in node.children:
        leaves.extend(get_leaf_nodes(child))
    
    return leaves