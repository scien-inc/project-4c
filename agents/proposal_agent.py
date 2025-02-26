"""
Proposal Agent for ROI calculation and proposal generation
"""
import os
from typing import Literal, Dict, List, Any, Optional, Tuple, cast

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from domain.schemas import ProposalState, ROICalculation, ProposalAnalysis
from domain.roitree import ROINode
from agents.prompts import PROPOSAL_ANALYSIS_PROMPT, PROPOSAL_REFLECTION_PROMPT, PROPOSAL_SYSTEM_PROMPT
from utils.parsers import parse_roi_calculation, parse_proposal_reflection
from utils.visualization import (
    format_tree_for_display,
    get_node_path_string,
    build_nodes_dictionary,
    format_roi_calculation
)

# Initialize LLM
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,  # Lower temperature for more consistent, focused responses
    streaming=True    # Enable streaming for real-time output
)
analysis_model = model
reflection_model = ChatOpenAI(model="gpt-4o", temperature=0.1)  # Even lower temp for reflection


# =========================
# Helper Functions
# =========================

def get_next_node_to_analyze(state: ProposalState) -> Optional[str]:
    """
    Find the next node that should be analyzed
    
    Strategy: 
    1. Start with leaf nodes
    2. Once all children of a parent are analyzed, analyze the parent
    
    Returns:
        Node ID of the next node to analyze, or None if all nodes are analyzed
    """
    if not state["root_node"]:
        return None
        
    nodes_dict = build_nodes_dictionary(state["root_node"])
    analyzed_nodes = set(state["analyzed_nodes"])
    
    # Helper function to check if a node and all its descendants are analyzed
    def is_subtree_analyzed(node: ROINode) -> bool:
        if node.node_id not in analyzed_nodes:
            return False
            
        return all(is_subtree_analyzed(child) for child in node.children)
    
    # Helper function to find next node to analyze
    def find_next_node(node: ROINode) -> Optional[str]:
        # If this node has no children (leaf node) and isn't analyzed, analyze it
        if not node.children and node.node_id not in analyzed_nodes:
            return node.node_id
            
        # If this node has children, first make sure all children are analyzed
        for child in node.children:
            if not is_subtree_analyzed(child):
                return find_next_node(child)
                
        # If all children are analyzed but this node isn't, analyze it
        if node.node_id not in analyzed_nodes:
            return node.node_id
            
        # Everything in this subtree is analyzed
        return None
    
    return find_next_node(state["root_node"])


def get_node_analysis_context(state: ProposalState, node_id: str) -> Dict[str, Any]:
    """
    Get the context needed for analyzing a specific node
    
    Returns:
        Dictionary with context information for the node
    """
    node = state["root_node"].find_node_by_id(node_id)
    if not node:
        return {
            "current_node_name": "Unknown Node",
            "current_node_details": "Node not found",
            "current_node_importance": 0,
            "child_nodes_description": "No children",
            "node_context": "Node not found in the ROI tree"
        }
    
    # Get node details
    current_node_name = node.name
    current_node_details = node.details or "No details available"
    current_node_importance = f"{node.importance_factor:.1%}"
    
    # Get child nodes description
    child_nodes_description = "No child nodes"
    if node.children:
        child_lines = []
        for child in node.children:
            child_line = f"- {child.name} (Importance: {child.importance_factor:.1%})"
            if child.details:
                child_line += f"\n  Details: {child.details}"
            child_lines.append(child_line)
        child_nodes_description = "\n".join(child_lines)
    
    # Get overall context in the tree
    nodes_dict = build_nodes_dictionary(state["root_node"])
    
    # Build a path to this node (simplified implementation)
    path = []
    current = node
    parent_map = {}
    
    # Build a map of nodes to their parents
    def build_parent_map(node: ROINode, parent: Optional[ROINode] = None):
        if parent:
            parent_map[node.node_id] = parent
        for child in node.children:
            build_parent_map(child, node)
    
    build_parent_map(state["root_node"])
    
    # Build the path
    current_id = node.node_id
    while current_id in parent_map:
        parent = parent_map[current_id]
        path.insert(0, parent.name)
        current_id = parent.node_id
    
    path.append(node.name)
    node_context = "Path: " + " > ".join(path)
    
    return {
        "current_node_name": current_node_name,
        "current_node_details": current_node_details,
        "current_node_importance": current_node_importance,
        "child_nodes_description": child_nodes_description,
        "node_context": node_context
    }


def update_node_values(state: ProposalState, calculations: List[ROICalculation]) -> ProposalState:
    """
    Update node values based on ROI calculations
    
    Args:
        state: Current state
        calculations: List of ROI calculations
        
    Returns:
        Updated state
    """
    if not calculations:
        return state
    
    # For each calculation, find the corresponding node and update its value
    nodes_dict = build_nodes_dictionary(state["root_node"])
    
    for calc in calculations:
        node_id = calc.node_id
        
        # If node_id is "auto", try to find node by name
        if node_id == "auto":
            for nid, node in nodes_dict.items():
                if node.name.lower() == calc.name.lower():
                    node_id = nid
                    break
        
        # Update the node if found
        if node_id in nodes_dict:
            node = nodes_dict[node_id]
            node.value = calc.estimated_value
            
            # Add to analyzed nodes
            if node_id not in state["analyzed_nodes"]:
                state["analyzed_nodes"].append(node_id)
        
        # Store calculation in the state
        if node_id not in state["roi_calculations"]:
            state["roi_calculations"][node_id] = {
                "name": calc.name,
                "value": calc.estimated_value,
                "confidence": calc.confidence,
                "assumptions": calc.assumptions
            }
    
    # Calculate ROI for the whole tree
    if state["root_node"]:
        roi_result = state["root_node"].calculate_roi()
        state["roi_calculations"]["summary"] = roi_result
    
    return state


# =========================
# Graph nodes (agent steps)
# =========================

def propose_solutions(state: ProposalState) -> ProposalState:
    """
    特定のノードのROIを分析し、ツリーを更新する
    """
    # 次に分析するノードを見つける
    next_node_id = state["current_node_id"] or get_next_node_to_analyze(state)
    
    if not next_node_id:
        # 分析するノードがもうない
        if "summary" not in state["roi_calculations"]:
            # 最終ROIを計算
            roi_result = state["root_node"].calculate_roi()
            state["roi_calculations"]["summary"] = roi_result
            
        # ROIをまとめるメッセージを作成
        summary = state["roi_calculations"].get("summary", {})
        summary_text = f"""
分析に基づいて、最終的なROIサマリーは以下のとおりです:

総合的な潜在価値: ¥{summary.get('weighted_value', 0)*1000:,.0f}

主要コンポーネント:
"""
        # 主要コンポーネントを追加
        for child in summary.get('children_values', []):
            summary_text += f"- {child['name']}: ¥{child['weighted_value']*1000:,.0f} (全体の{child['importance_factor']:.1%})\n"
        
        state["messages"].append(
            SystemMessage(content=f"すべてのノードの分析が完了しました。{summary_text}")
        )
        
        return state
    
    # 現在のノードを更新
    state["current_node_id"] = next_node_id
    
    # このノードのコンテキストを取得
    context = get_node_analysis_context(state, next_node_id)
    
    # 明示的に日本語で応答するよう指示を追加
    system_message = SystemMessage(content=PROPOSAL_SYSTEM_PROMPT + "\n\n特に重要: すべての応答は必ず日本語で行ってください。")
    
    messages = state["messages"] + [
        HumanMessage(content=f"""
ROIツリーの以下のノードについてROIを推定しましょう:

ノード: {context["current_node_name"]}
説明: {context["current_node_details"]}
重要度係数: {context["current_node_importance"]}

子ノード（存在する場合）:
{context["child_nodes_description"]}

ROIツリー全体におけるコンテキスト:
{context["node_context"]}

このコンポーネントの潜在的価値を見積もるのを手伝ってください。十分な情報に基づいた見積もりをするために必要な質問をするか、利用可能な情報に基づいて推奨事項を提供してください。

各見積もりについて、以下を提供してください:
1. 見積もり価値（金銭的な観点で）
2. 信頼度レベル（0-100%）
3. 見積もりの背後にある主要な前提条件
4. 時間的考慮（一回限りか継続的か、いつ利益が実現されるか）

必ず日本語で回答してください。
""")
    ]
    
    # このノードの分析を生成
    response = analysis_model.invoke([system_message] + messages)
    
    # メッセージを更新
    state["messages"].append(
        HumanMessage(content=f"「{context['current_node_name']}」のROIを分析しましょう。")
    )
    state["messages"].append(response)
    
    # 応答を解析してROI計算を抽出
    calculations = parse_roi_calculation(response)
    
    if calculations:
        # 計算のノードIDを更新
        for calc in calculations:
            if calc.node_id == "auto":
                calc.node_id = next_node_id
        
        # ツリーのROI値を更新
        state = update_node_values(state, calculations)
    
    return state


def self_reflect_proposal(state: ProposalState) -> ProposalState:
    """
    ROI分析が完了し十分かどうかを評価する
    """
    # 自己反省のコンテキストを準備
    nodes_dict = build_nodes_dictionary(state["root_node"])
    total_nodes = len(nodes_dict)
    analyzed_nodes_count = len(state["analyzed_nodes"])
    
    # 見積りが不足しているノードをカウント
    missing_estimates = []
    for node_id, node in nodes_dict.items():
        if node.value is None:
            missing_estimates.append(node.name)
    
    missing_estimates_count = len(missing_estimates)
    
    # ROIサマリーを整形
    roi_summary = "まだROI計算は実行されていません"
    if "summary" in state["roi_calculations"]:
        summary = state["roi_calculations"]["summary"]
        roi_summary = format_roi_calculation(summary)
    
    # 反省プロンプトを生成
    reflection_input = PROPOSAL_REFLECTION_PROMPT.format(
        messages=state["messages"][-5:],  # コンテキストには最近のメッセージのみ使用
        roi_calculation_summary=roi_summary,
        analyzed_nodes_count=analyzed_nodes_count,
        total_nodes_count=total_nodes,
        missing_estimates_count=missing_estimates_count
    )
    
    try:
        # LLMから反省を取得
        reflection_response = reflection_model.invoke(reflection_input)
        
        # 反省を解析
        success, analysis = parse_proposal_reflection(reflection_response)
        
        if success and analysis:
            state["self_reflection"] = analysis
            state["proposal_complete"] = analysis.proposal_complete
        else:
            # 解析に失敗した場合のフォールバック
            state["iteration_count"] += 1
            state["proposal_complete"] = state["iteration_count"] >= state["max_iterations"]
    except Exception as e:
        print(f"自己反省中にエラーが発生しました: {str(e)}")
        # エラーの場合のフォールバック
        state["iteration_count"] += 1
        state["proposal_complete"] = state["iteration_count"] >= state["max_iterations"]
    
    return state


def should_continue_proposal(state: ProposalState) -> Literal["continue", "__end__"]:
    """Decide whether to continue analysis or end"""
    return "continue" if not state["proposal_complete"] else "__end__"


# =========================
# Graph construction
# =========================

def build_proposal_graph() -> Any:  # Using Any for the return type to avoid circular imports
    """Build and compile the proposal agent graph"""
    graph = StateGraph(ProposalState)
    
    # Add nodes
    graph.add_node("propose_solutions", propose_solutions)
    graph.add_node("self_reflect_proposal", self_reflect_proposal)
    
    # Add edges
    graph.add_edge("propose_solutions", "self_reflect_proposal")
    graph.add_conditional_edges(
        "self_reflect_proposal",
        should_continue_proposal,
        {
            "continue": "propose_solutions",
            "__end__": END
        }
    )
    
    graph.set_entry_point("propose_solutions")
    
    return graph.compile()


def init_proposal_state(root_node: ROINode) -> ProposalState:
    """Initialize the proposal state"""
    return ProposalState(
        messages=[],
        root_node=root_node,
        roi_calculations={},
        current_node_id=None,
        analyzed_nodes=[],
        iteration_count=0,
        max_iterations=10,
        self_reflection=None,
        proposal_complete=False
    )