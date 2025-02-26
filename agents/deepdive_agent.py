"""
Deepdive Agent for ROI tree exploration
"""
import os
from typing import Literal, Dict, List, Any, Optional, Tuple, cast

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from domain.schemas import DeepdiveState, ROINodeUpdate, ROIAnalysis, NodePath
from domain.roitree import ROINode, create_default_roi_tree
from agents.prompts import DEEPDIVE_EXPLORATION_PROMPT, DEEPDIVE_REFLECTION_PROMPT, DEEPDIVE_SYSTEM_PROMPT
from utils.parsers import parse_roi_node_update, parse_reflection_analysis
from utils.visualization import (
    format_tree_for_display, 
    get_node_path_string,
    generate_mermaid_diagram,
    build_nodes_dictionary,
    get_tree_statistics
)

# Initialize LLM
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2,  # Lower temperature for more consistent, focused responses
    streaming=True    # Enable streaming for real-time output
)
exploration_model = model
reflection_model = ChatOpenAI(model="gpt-4o", temperature=0.1)  # Even lower temp for reflection


# =========================
# Helper Functions
# =========================

def find_node_by_path(root_node: ROINode, node_path: NodePath) -> Optional[ROINode]:
    """Find a node using the path from root"""
    if not node_path:
        return root_node
        
    current_node = root_node
    for node_id in node_path:
        found = current_node.find_node_by_id(node_id)
        if found:
            current_node = found
        else:
            return None
            
    return current_node


def get_tree_context(state: DeepdiveState) -> str:
    """Get formatted context of the current ROI tree for prompts"""
    if not state["root_node"]:
        return "No ROI tree has been created yet."
        
    # Format the full tree for display
    tree_text = format_tree_for_display(state["root_node"])
    
    # Get information about the current node and path
    nodes_dict = build_nodes_dictionary(state["root_node"])
    path_text = get_node_path_string(state["node_path"], nodes_dict)
    
    return f"""ROI Tree Structure:
{tree_text}

Current Path: {path_text}
"""


def update_roi_tree(
    state: DeepdiveState, 
    updates: List[ROINodeUpdate]
) -> Tuple[DeepdiveState, List[str]]:
    """
    Update the ROI tree based on parsed updates
    
    Args:
        state: Current state
        updates: List of node updates
        
    Returns:
        Updated state and list of update descriptions
    """
    if not state["root_node"]:
        # Initialize with default tree if none exists
        state["root_node"] = create_default_roi_tree()
        if not state["current_node_id"]:
            state["current_node_id"] = state["root_node"].node_id
            state["node_path"] = [state["root_node"].node_id]
    
    # Find the current node
    current_node = None
    if state["current_node_id"]:
        current_node = find_node_by_path(state["root_node"], state["node_path"])
    
    if not current_node:
        current_node = state["root_node"]
        state["current_node_id"] = current_node.node_id
        state["node_path"] = [current_node.node_id]
    
    # Apply updates
    update_descriptions = []
    
    for update in updates:
        parent_node = current_node
        
        # If a parent node ID is specified, try to find it
        if update.parent_node_id:
            parent_search = state["root_node"].find_node_by_id(update.parent_node_id)
            if parent_search:
                parent_node = parent_search
        
        # Create and add the new node
        new_node = ROINode(
            name=update.name,
            details=update.details,
            importance_factor=update.importance_factor,
            value=update.value
        )
        
        parent_node.add_child(new_node)
        
        # Update descriptions for user feedback
        update_descriptions.append(
            f"Added node '{new_node.name}' under '{parent_node.name}' "
            f"with importance factor {new_node.importance_factor:.1%}"
        )
        
        # Update the current focus to this new node
        state["current_node_id"] = new_node.node_id
        
        # Update the node path
        if parent_node.node_id in state["node_path"]:
            # Find the position of the parent in the path
            idx = state["node_path"].index(parent_node.node_id)
            # Truncate the path to the parent and add the new node
            state["node_path"] = state["node_path"][:idx+1] + [new_node.node_id]
        else:
            # Add the new node to the path
            state["node_path"].append(new_node.node_id)
    
    # Normalize importance factors
    state["root_node"].normalize_importance_factors()
    
    # Add current node to exploration history if not already there
    if state["current_node_id"] not in state["exploration_history"]:
        state["exploration_history"].append(state["current_node_id"])
    
    return state, update_descriptions


# =========================
# Graph nodes (agent steps)
# =========================

def deepdive_conversation(state: DeepdiveState) -> DeepdiveState:
    """
    ユーザーと対話し、ROIツリー探索を行う
    """
    # 現在のノード情報を取得
    current_node = None
    current_node_name = "ルート"
    current_node_details = "トップレベルのROIコンポーネント"
    
    if state["current_node_id"]:
        current_node = find_node_by_path(state["root_node"], state["node_path"])
        if current_node:
            current_node_name = current_node.name
            current_node_details = current_node.details or "詳細情報なし"
    
    # プロンプト用のツリーコンテキストを取得
    tree_context = get_tree_context(state)
    
    # 探索プロンプトを使用して応答を生成
    # 明示的に日本語で応答するよう指示を追加
    system_message = SystemMessage(content=DEEPDIVE_SYSTEM_PROMPT + "\n\n特に重要: すべての応答は必ず日本語で行ってください。")
    
    messages = state["messages"] + [
        HumanMessage(content=f"""
現在のROIツリーコンテキスト:
{tree_context}

現在のフォーカス: {current_node_name}
説明: {current_node_details}

このROIツリーの側面をさらに探索するのを手伝ってください。サブコンポーネントを特定するための的を絞った質問をするか、適切な場合は新しいブランチを提案してください。

重要度係数について考えることを忘れないでください - 各サブコンポーネントは兄弟コンポーネントと比較してどの程度重要ですか？重要度係数は兄弟間で合計100%になるようにしてください。

必ず日本語で回答してください。
""")
    ]
    
    response = exploration_model.invoke([system_message] + messages)
    
    # メッセージを更新
    state["messages"].append(
        HumanMessage(content=f"「{current_node_name}」についてさらに詳しく教えてください。")
    )
    state["messages"].append(response)
    
    # 応答を解析してノード更新を抽出
    updates = parse_roi_node_update(response)
    
    if updates:
        # ROIツリーに更新を適用
        state, update_descriptions = update_roi_tree(state, updates)
    
    return state


def self_reflect_deepdive(state: DeepdiveState) -> DeepdiveState:
    """
    LLMを使用してROIツリー探索が十分かどうかを評価する
    """
    # 自己反省のコンテキストを準備
    tree_stats = get_tree_statistics(state["root_node"])
    
    # 完全なツリー表現を取得
    full_tree = format_tree_for_display(state["root_node"])
    mermaid_diagram = generate_mermaid_diagram(state["root_node"])
    
    # 探索履歴を読みやすいフォーマットに変換
    nodes_dict = build_nodes_dictionary(state["root_node"])
    exploration_history = []
    
    for node_id in state["exploration_history"]:
        if node_id in nodes_dict:
            exploration_history.append(nodes_dict[node_id].name)
    
    # 反省プロンプトを生成
    reflection_input = DEEPDIVE_REFLECTION_PROMPT.format(
        messages=state["messages"][-5:],  # コンテキストには最近のメッセージのみ使用
        full_tree_representation=full_tree,
        exploration_history=", ".join(exploration_history),
        total_nodes=tree_stats["total_nodes"],
        max_depth=tree_stats["max_depth"],
        min_nodes_per_branch=state["min_nodes_per_branch"],
        shallow_branches=", ".join(tree_stats["shallow_branches"])
    )
    
    try:
        # LLMから反省を取得
        reflection_response = reflection_model.invoke(reflection_input)
        
        # 反省を解析
        success, analysis = parse_reflection_analysis(reflection_response)
        
        if success and analysis:
            state["self_reflection"] = analysis
            state["exploration_complete"] = not analysis.deepdive_needed
            
            # フォーカスが提案されており、探索が完了していない場合、そのノードに移動を試みる
            if analysis.suggested_focus and not state["exploration_complete"]:
                # まず名前でノードを検索（簡易実装）
                nodes_dict = build_nodes_dictionary(state["root_node"])
                for node_id, node in nodes_dict.items():
                    if node.name.lower() == analysis.suggested_focus.lower():
                        state["current_node_id"] = node_id
                        # このノードへのパスを再構築（簡易化）
                        state["node_path"] = [state["root_node"].node_id, node_id]
                        break
        else:
            # 解析に失敗した場合のフォールバック
            state["iteration_count"] += 1
            state["exploration_complete"] = state["iteration_count"] >= state["max_iterations"]
    except Exception as e:
        print(f"自己反省中にエラーが発生しました: {str(e)}")
        # エラーの場合のフォールバック
        state["iteration_count"] += 1
        state["exploration_complete"] = state["iteration_count"] >= state["max_iterations"]
    
    return state


def should_continue_deepdive(state: DeepdiveState) -> Literal["continue", "__end__"]:
    """Decide whether to continue exploration or end"""
    return "continue" if not state["exploration_complete"] else "__end__"


# =========================
# Graph construction
# =========================

def build_deepdive_graph() -> Any:  # Using Any for the return type to avoid circular imports
    """Build and compile the deepdive agent graph"""
    graph = StateGraph(DeepdiveState)
    
    # Add nodes
    graph.add_node("deepdive_conversation", deepdive_conversation)
    graph.add_node("self_reflect_deepdive", self_reflect_deepdive)
    
    # Add edges
    graph.add_edge("deepdive_conversation", "self_reflect_deepdive")
    graph.add_conditional_edges(
        "self_reflect_deepdive",
        should_continue_deepdive,
        {
            "continue": "deepdive_conversation",
            "__end__": END
        }
    )
    
    graph.set_entry_point("deepdive_conversation")
    
    return graph.compile()


def init_deepdive_state() -> DeepdiveState:
    """Initialize the deepdive state"""
    return DeepdiveState(
        messages=[],
        root_node=create_default_roi_tree(),
        current_node_id=None,
        node_path=[],
        exploration_history=[],
        iteration_count=0,
        max_iterations=10,
        min_nodes_per_branch=3,
        self_reflection=None,
        exploration_complete=False
    )