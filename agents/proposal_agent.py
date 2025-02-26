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
from domain.reflection import ReflectionManager, TaskReflector, format_reflections
from agents.prompts import PROPOSAL_SYSTEM_PROMPT, PROPOSAL_REFLECTION_PROMPT
from utils.parsers import parse_roi_calculation, parse_proposal_reflection
from utils.visualization import (
    format_tree_for_display,
    get_node_path_string,
    build_nodes_dictionary,
    format_roi_calculation
)

# 反省管理の共有インスタンスを使用
from agents.deepdive_agent import reflection_manager

# Initialize LLM
model = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,  # Lower temperature for more consistent, focused responses
    streaming=True    # Enable streaming for real-time output
)
analysis_model = model
reflection_model = ChatOpenAI(model="gpt-4", temperature=0.1)  # Even lower temp for reflection

# タスク反省機能を初期化
task_reflector = TaskReflector(llm=reflection_model, reflection_manager=reflection_manager)


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
    ユーザーの入力に基づいてノードのROIを分析（自動会話生成なし）
    """
    # 次に分析するノードを見つける
    next_node_id = state["current_node_id"] or get_next_node_to_analyze(state)
    
    if not next_node_id:
        # No more nodes to analyze
        if "summary" not in state["roi_calculations"]:
            # Calculate final ROI
            roi_result = state["root_node"].calculate_roi()
            state["roi_calculations"]["summary"] = roi_result
            
        # Create a message summarizing the ROI
        summary = state["roi_calculations"].get("summary", {})
        summary_text = f"""
分析に基づいて、最終的なROIサマリーは以下のとおりです:

総合的な潜在価値: ¥{summary.get('weighted_value', 0)*1000:,.0f}

主要コンポーネント:
"""
        # Add major components
        for child in summary.get('children_values', []):
            summary_text += f"- {child['name']}: ¥{child['weighted_value']*1000:,.0f} (全体の{child['importance_factor']:.1%})\n"
        
        state["messages"].append(
            SystemMessage(content=f"すべてのノードの分析が完了しました。{summary_text}")
        )
        
        return state
    
    # Update current node
    state["current_node_id"] = next_node_id
    
    # Get context for this node
    context = get_node_analysis_context(state, next_node_id)
    
    # 関連する過去のリフレクションを取得
    relevant_reflections = reflection_manager.get_relevant_reflections(
        f"{context['current_node_name']} ROI計算"
    )
    reflection_text = format_reflections(relevant_reflections)
    
    # 最新のユーザーメッセージを取得
    latest_user_message = None
    for msg in reversed(state["messages"]):
        if msg.type == "human":
            latest_user_message = msg.content
            break
    
    if not latest_user_message:
        # 初回の場合や、ユーザーメッセージがない場合はデフォルトの応答を返す
        response_content = f"""
現在、「{context['current_node_name']}」ノードのROI分析を行っています。
重要度係数: {context['current_node_importance']}

このノードについて、価値見積もりに必要な情報をお聞かせください。
例えば、予想される効果の大きさや、実現可能性などについて教えていただけると助かります。
        """
        response = AIMessage(content=response_content)
    else:
        # システムメッセージを設定
        system_message = SystemMessage(content=PROPOSAL_SYSTEM_PROMPT + 
            f"\n\n以下の過去のリフレクションを考慮してください:\n{reflection_text}\n\n" +
            "特に重要: すべての応答は必ず日本語で行ってください。")
        
        messages = state["messages"][-5:] + [
            HumanMessage(content=f"""
ROIツリーの以下のノードについてROIを推定しましょう:

ノード: {context["current_node_name"]}
説明: {context["current_node_details"]}
重要度係数: {context["current_node_importance"]}

子ノード（存在する場合）:
{context["child_nodes_description"]}

ROIツリー全体におけるコンテキスト:
{context["node_context"]}

ユーザーからの質問/入力: {latest_user_message}

このノードに関するユーザーの質問に回答するか、より詳細な情報を求めてください。

各見積もりについて、以下の情報を収集することを目指してください:
1. 見積もり価値（金銭的な観点で）
2. 信頼度レベル（0-100%）
3. 見積もりの背後にある主要な前提条件
4. 時間的考慮（一回限りか継続的か、いつ利益が実現されるか）

過去のリフレクションに基づいた改善点も考慮してください。

必ず日本語で回答してください。
""")
        ]
        
        response = analysis_model.invoke([system_message] + messages)
    
    # 応答をメッセージリストに追加（ユーザーメッセージは既に追加されているはず）
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
    
    # タスクと結果に対する反省を実行
    task_context = {
        "node_name": context["current_node_name"],
        "node_details": context["current_node_details"],
        "node_importance": context["current_node_importance"],
        "calculation_result": calculations[0].estimated_value if calculations else None,
        "confidence": calculations[0].confidence if calculations else None
    }
    
    task_reflector.run(
        task=f"ROIノード「{context['current_node_name']}」の価値見積もり",
        result=response.content,
        context=task_context
    )
    
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
    
    # 関連する過去のリフレクションを取得
    relevant_reflections = reflection_manager.get_relevant_reflections("ROI提案の完了判断")
    reflection_context = format_reflections(relevant_reflections)
    
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
        reflection_system_message = SystemMessage(content=f"""あなたはROI計算の完全性を評価する専門家です。
あなたの仕事は、ROI提案を最終決定するのに十分な情報があるかどうかを判断することです。

完全なROI計算は以下の特徴を持つべきです:
1. すべての主要コンポーネントの見積もり値がある
2. 適切な信頼度レベルが含まれている
3. 主要な前提条件が文書化されている
4. コストと利益の両方を考慮している
5. 時間枠を考慮している

以下の過去のリフレクションも考慮してください:
{reflection_context}

正確な単一行のJSON形式で応答してください。改行を含めないでください。
""")
        
        reflection_messages = [
            reflection_system_message,
            HumanMessage(content=reflection_input)
        ]
        
        reflection_response = reflection_model.invoke(reflection_messages)
        
        # 反省を解析
        success, analysis = parse_proposal_reflection(reflection_response)
        
        if success and analysis:
            state["self_reflection"] = analysis
            state["proposal_complete"] = analysis.proposal_complete
            
            # リフレクション結果を記録
            task_reflector.run(
                task="ROI提案の完了判断",
                result=f"完了: {analysis.proposal_complete}, 信頼度: {analysis.roi_confidence}, 理由: {analysis.reason}",
                context={
                    "analyzed_nodes": analyzed_nodes_count,
                    "total_nodes": total_nodes,
                    "missing_estimates_count": missing_estimates_count,
                    "roi_confidence": analysis.roi_confidence,
                    "missing_information": analysis.missing_information
                }
            )
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