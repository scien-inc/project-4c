"""
utils/visualization.py
Visualization utilities for ROI tree analysis
"""
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import numpy as np
import re


def render_mermaid(mermaid_syntax, height=500):
    """
    Render a Mermaid diagram in Streamlit using HTML and JavaScript
    
    Args:
        mermaid_syntax: Mermaid diagram syntax
        height: Height of the diagram in pixels
    """
    # Clean Mermaid syntax (remove leading and trailing whitespace)
    mermaid_syntax = mermaid_syntax.strip()
    
    html = f"""
    <script src="https://cdn.jsdelivr.net/npm/mermaid@9.3.0/dist/mermaid.min.js"></script>
    <div class="mermaid" style="width: 100%;">
    {mermaid_syntax}
    </div>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            flowchart: {{ 
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
    """
    components.html(html, height=height)


def render_combined_trees(challenge_mermaid, proposal_mermaid, height=800):
    """
    Render combined challenge and proposal trees
    
    Args:
        challenge_mermaid: Mermaid diagram of the challenge tree (top-down)
        proposal_mermaid: Mermaid diagram of the proposal tree (bottom-up)
        height: Height of the diagram in pixels
    """
    # Remove flowchart directive from challenge tree
    challenge_lines = challenge_mermaid.strip().split('\n')
    if challenge_lines[0].startswith('graph ') or challenge_lines[0].startswith('flowchart '):
        challenge_lines = challenge_lines[1:]
    challenge_content = '\n'.join(challenge_lines)
    
    # Remove flowchart directive from proposal tree
    proposal_lines = proposal_mermaid.strip().split('\n')
    if proposal_lines[0].startswith('graph ') or proposal_lines[0].startswith('flowchart '):
        proposal_lines = proposal_lines[1:]
    proposal_content = '\n'.join(proposal_lines)
    
    # Extract leaf nodes from challenge tree
    challenge_leaf_nodes = extract_leaf_nodes(challenge_mermaid)
    
    # Extract leaf nodes from proposal tree
    proposal_nodes = extract_leaf_nodes(proposal_mermaid)
    
    # Generate connections between leaf nodes
    connections = []
    
    # If number of leaf nodes matches, connect them one-to-one
    if len(challenge_leaf_nodes) == len(proposal_nodes):
        for i in range(len(challenge_leaf_nodes)):
            challenge_node = challenge_leaf_nodes[i][0]
            proposal_node = proposal_nodes[i][0]
            # Connect challenge node to proposal node
            connections.append(f"{challenge_node} -.-> {proposal_node}")
    else:
        # If numbers don't match, connect as many as possible
        max_connections = min(len(challenge_leaf_nodes), len(proposal_nodes))
        for i in range(max_connections):
            challenge_node = challenge_leaf_nodes[i][0]
            proposal_node = proposal_nodes[i][0]
            connections.append(f"{challenge_node} -.-> {proposal_node}")
    
    # Create combined Mermaid diagram
    combined_mermaid = f"""
flowchart TD
    %% Style definitions
    classDef challenge fill:#f9f9f9,stroke:#333,stroke-width:1px
    classDef proposal fill:#e6f7ff,stroke:#0066cc,stroke-width:1px
    classDef connection stroke:#999,stroke-width:1px,stroke-dasharray: 5 5
    
    %% Challenge tree (top-down)
    subgraph 課題分析["課題分析"]
    {challenge_content}
    end
    
    %% Proposal tree (bottom-up)
    subgraph 提案策["提案策"]
    {proposal_content}
    end
    
    %% Connect challenge leaf nodes to proposal nodes
    {'\n    '.join(connections)}
    
    %% Apply styles
    class 課題分析 challenge
    class 提案策 proposal
    """
    
    render_mermaid(combined_mermaid, height=height)


def render_2d_mapping(mapping_result, x_axis, y_axis, height=600):
    """
    Render 2D mapping of ROI nodes
    
    Args:
        mapping_result: MappingResult object (Pydantic model)
        x_axis: X-axis label
        y_axis: Y-axis label
        height: Height of the figure in pixels
        
    Returns:
        Matplotlib figure or None if no mapping data
    """
    # MappingResultはPydanticモデルなので、直接属性にアクセスする
    # mapping_data = mapping_result.get("mapping", [])  # 修正前
    mapping_data = mapping_result.mapping if hasattr(mapping_result, 'mapping') else []  # 修正後
    
    if not mapping_data:
        return None
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract values
    x_values = [item.x_value for item in mapping_data]
    y_values = [item.y_value for item in mapping_data]
    node_names = [item.name for item in mapping_data]
    
    # Create color map based on priority
    colors = np.linspace(0, 1, len(x_values))
    
    # Plot points
    scatter = ax.scatter(x_values, y_values, s=120, c=colors, cmap='viridis', alpha=0.8, edgecolors='white')
    
    # Add labels
    for i, name in enumerate(node_names):
        short_name = name[:15] + "..." if len(name) > 15 else name
        ax.annotate(short_name, (x_values[i], y_values[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # Set axis labels and limits
    ax.set_xlabel(x_axis, fontsize=12)
    ax.set_ylabel(y_axis, fontsize=12)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Divide into quadrants
    ax.axhline(y=5, color='r', linestyle='-', alpha=0.3)
    ax.axvline(x=5, color='r', linestyle='-', alpha=0.3)
    
    # Add quadrant labels
    ax.text(2.5, 7.5, f"低{x_axis}\n高{y_axis}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(7.5, 7.5, f"高{x_axis}\n高{y_axis}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(2.5, 2.5, f"低{x_axis}\n低{y_axis}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    ax.text(7.5, 2.5, f"高{x_axis}\n低{y_axis}", ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
    
    # Set title
    plt.title(f"ROIノード2次元マッピング: {x_axis} vs {y_axis}", fontsize=14)
    
    # Add tight layout
    plt.tight_layout()
    
    return fig


def extract_leaf_nodes(mermaid_code):
    """
    Extract leaf nodes (nodes without children) from Mermaid code
    
    Args:
        mermaid_code: Mermaid diagram code
        
    Returns:
        List of tuples (node_id, node_label)
    """
    all_node_ids = set()
    parent_node_ids = set()
    node_definitions = {}
    lines = mermaid_code.strip().split('\n')
    
    # First capture node definitions
    for line in lines:
        # Capture node definitions (e.g., node1["Text"])
        matches = re.findall(r'(\w+)\[\"([^\"]+)\"', line)
        for match in matches:
            node_id = match[0]
            node_label = match[1]
            all_node_ids.add(node_id)
            node_definitions[node_id] = node_label
    
    # Then find parent nodes
    for line in lines:
        # Capture edge definitions (e.g., node1 --> node2)
        edge_matches = re.findall(r'(\w+)\s*-->', line)
        for match in edge_matches:
            parent_node_ids.add(match)
    
    # Leaf nodes are nodes that are not parents
    leaf_nodes = all_node_ids - parent_node_ids
    
    # Return leaf nodes and their labels
    result = [(node_id, node_definitions.get(node_id, "")) for node_id in leaf_nodes]
    return result


def has_numerical_value(node_label: str) -> bool:
    """
    Check if a node label contains numerical information (amount, percentage, etc.)
    
    Args:
        node_label: Node label text
        
    Returns:
        True if the node label contains numerical information, False otherwise
    """
    # Amount pattern (e.g., 1億円, 500万円, 1,000円)
    amount_pattern = r'(\d+[,\.]?\d*\s*[億万千]?円)'
    # Percentage pattern (e.g., 25%, 3.5%)
    percentage_pattern = r'(\d+[,\.]?\d*\s*%)'
    
    return bool(re.search(amount_pattern, node_label)) or bool(re.search(percentage_pattern, node_label))


def extract_nodes_without_values(mermaid_code):
    """
    Extract leaf nodes without numerical values
    
    Args:
        mermaid_code: Mermaid diagram code
        
    Returns:
        List of tuples (node_id, node_label) of leaf nodes without numerical values
    """
    leaf_nodes = extract_leaf_nodes(mermaid_code)
    nodes_without_values = []
    
    for node_id, node_label in leaf_nodes:
        if not has_numerical_value(node_label):
            nodes_without_values.append((node_id, node_label))
    
    return nodes_without_values


def update_node_value_in_mermaid(mermaid_code, node_id, new_value):
    """
    Update or add a value to a node in Mermaid code
    
    Args:
        mermaid_code: Mermaid diagram code
        node_id: ID of the node to update
        new_value: New value to add or update
        
    Returns:
        Updated Mermaid diagram code
    """
    lines = mermaid_code.strip().split('\n')
    updated_lines = []
    
    for line in lines:
        # Search for node definition (e.g., node1["Text"])
        node_match = re.search(rf'({node_id}\[\")([^\"]+)(\"\])', line)
        if node_match:
            # Check for existing value
            current_label = node_match.group(2)
            value_pattern = r'\(([^)]+)\)'
            value_match = re.search(value_pattern, current_label)
            
            if value_match:
                # Update existing value
                updated_label = re.sub(value_pattern, f'({new_value})', current_label)
            else:
                # Add new value
                updated_label = f"{current_label} ({new_value})"
            
            # Update line
            updated_line = f'{node_match.group(1)}{updated_label}{node_match.group(3)}'
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)