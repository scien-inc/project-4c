"""
Utilities for visualizing ROI trees and calculations
"""
from typing import Any, Dict, List, Optional
import textwrap


def format_tree_for_display(node: Any, indent: int = 0, show_details: bool = True) -> str:
    """
    Format a ROI tree node and its children for text display
    
    Args:
        node: ROI tree node
        indent: Current indentation level
        show_details: Whether to show node details
        
    Returns:
        Formatted string representation
    """
    prefix = "  " * indent
    lines = [f"{prefix}• **{node.name}**"]
    
    if show_details and node.details:
        wrapped_details = textwrap.fill(
            node.details, 
            width=80, 
            initial_indent=f"{prefix}  ", 
            subsequent_indent=f"{prefix}  "
        )
        lines.append(wrapped_details)
    
    # Add importance factor if not 1.0
    if node.importance_factor != 1.0:
        lines.append(f"{prefix}  Importance: {node.importance_factor:.1%}")
    
    # Add value if present
    if node.value is not None:
        lines.append(f"{prefix}  Value: {node.value:,.2f}")
    
    # Process children
    for child in node.children:
        child_text = format_tree_for_display(child, indent + 1, show_details)
        lines.append(child_text)
    
    return "\n".join(lines)


def get_node_path_string(node_path: List[str], nodes_dict: Dict[str, Any]) -> str:
    """
    Convert a node path to a readable string
    
    Args:
        node_path: List of node IDs in the path
        nodes_dict: Dictionary mapping node IDs to node objects
        
    Returns:
        Formatted path string
    """
    if not node_path:
        return "Root"
    
    path_names = []
    for node_id in node_path:
        if node_id in nodes_dict:
            path_names.append(nodes_dict[node_id].name)
        else:
            path_names.append("Unknown")
    
    return " > ".join(path_names)


def generate_mermaid_diagram(node: Any) -> str:
    """
    Generate a Mermaid diagram for an ROI tree
    
    Args:
        node: Root node of the ROI tree
        
    Returns:
        Mermaid diagram string
    """
    return node.get_full_mermaid()


def format_roi_calculation(roi_calc: Dict[str, Any], indent: int = 0) -> str:
    """
    ROI計算結果を表示用にフォーマット
    
    Args:
        roi_calc: ROI計算辞書
        indent: インデントレベル
        
    Returns:
        フォーマットされた文字列
    """
    prefix = "  " * indent
    lines = [f"{prefix}• **{roi_calc['name']}**"]
    
    # 重み付けされた値を追加
    lines.append(f"{prefix}  合計価値: ¥{roi_calc['weighted_value']*1000:,.0f}")
    
    # 子要素を処理
    if 'children_values' in roi_calc:
        for child in roi_calc['children_values']:
            # 重要度係数付きで子情報を追加
            child_value = child.get('weighted_value', 0)
            importance = child.get('importance_factor', 1.0)
            
            lines.append(f"{prefix}  • {child['name']}: ¥{child_value*1000:,.0f} (重み: {importance:.1%})")
            
            # 孫要素を再帰的に処理
            if 'children' in child and child['children']:
                child_calc = {
                    'name': child['name'],
                    'weighted_value': child['weighted_value'],
                    'children_values': child['children']
                }
                child_text = format_roi_calculation(child_calc, indent + 2)
                lines.append(child_text)
    
    return "\n".join(lines)


def build_nodes_dictionary(root_node: Any) -> Dict[str, Any]:
    """
    Build a dictionary mapping node IDs to nodes for quick lookup
    
    Args:
        root_node: Root node of the ROI tree
        
    Returns:
        Dictionary mapping node IDs to nodes
    """
    nodes_dict = {}
    
    def _traverse(node):
        nodes_dict[node.node_id] = node
        for child in node.children:
            _traverse(child)
    
    _traverse(root_node)
    return nodes_dict


def get_tree_statistics(root_node: Any) -> Dict[str, Any]:
    """
    Calculate statistics about the ROI tree
    
    Args:
        root_node: Root node of the ROI tree
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_nodes": 0,
        "max_depth": 0,
        "leaf_nodes": 0,
        "shallow_branches": []
    }
    
    def _traverse(node, depth=0):
        stats["total_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if not node.children:
            stats["leaf_nodes"] += 1
            if depth < 3:  # Consider branches with depth < 3 as shallow
                stats["shallow_branches"].append(node.name)
        
        for child in node.children:
            _traverse(child, depth + 1)
    
    _traverse(root_node)
    return stats