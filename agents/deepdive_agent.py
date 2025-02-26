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
from domain.reflection import ReflectionManager, TaskReflector, format_reflections
from agents.prompts import DEEPDIVE_SYSTEM_PROMPT, DEEPDIVE_REFLECTION_PROMPT
from utils.parsers import parse_roi_node_update, parse_reflection_analysis
from utils.visualization import (
    format_tree_for_display, 
    get_node_path_string,
    generate_mermaid_diagram,
    build_nodes_dictionary,
    get_tree_statistics
)

# 反省管理とタスク反省機能を初期化
# 自動リセット機能はStreamlitのセッションからmain.pyで設定される
reflection_manager = ReflectionManager()

# LLMの初期化
model = ChatOpenAI(
    model="gpt-4",
    temperature=0.2,  # 一貫性の高い応答のための低い温度
    streaming=True    # リアルタイム出力のためのストリーミング
)
exploration_model = model
reflection_model = ChatOpenAI(model="gpt-4", temperature=0.1)  # 反省にはさらに低い温度

# タスク反省機能を初期化
task_reflector = TaskReflector(llm=reflection_model, reflection_manager=reflection_manager)


# =========================
# ヘルパー関数
# =========================

def find_node_by_path(root_node: ROINode, node_path: NodePath) -> Optional[ROINode]:
    """パスからノードを見つける"""
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
    """プロンプト用の現在のROIツリーコンテキストを取得"""
    if not state["root_node"]:
        return "まだROIツリーが作成されていません。"
        
    # ツリーを表示用にフォーマット
    tree_text = format_tree_for_display(state["root_node"])
    
    # 現在のノードとパスの情報を取得
    nodes_dict = build_nodes_dictionary(state["root_node"])
    path_text = get_node_path_string(state["node_path"], nodes_dict)
    
    return f"""ROIツリー構造:
{tree_text}

現在のパス: {path_text}
"""


def update_roi_tree(
    state: DeepdiveState, 
    updates: List[ROINodeUpdate]
) -> Tuple[DeepdiveState, List[str]]:
    """
    解析された更新情報に基づいてROIツリーを更新
    
    Args:
        state: 現在の状態
        updates: ノード更新のリスト
        
    Returns:
        更新された状態と更新説明のリスト
    """
    if not state["root_node"]:
        # ツリーが存在しない場合、デフォルトツリーで初期化
        state["root_node"] = create_default_roi_tree()
        if not state["current_node_id"]:
            state["current_node_id"] = state["root_node"].node_id
            state["node_path"] = [state["root_node"].node_id]
    
    # 現在のノードを検索
    current_node = None
    if state["current_node_id"]:
        current_node = find_node_by_path(state["root_node"], state["node_path"])
    
    if not current_node:
        current_node = state["root_node"]
        state["current_node_id"] = current_node.node_id
        state["node_path"] = [current_node.node_id]
    
    # 更新を適用
    update_descriptions = []
    
    for update in updates:
        parent_node = current_node
        
        # 親ノードIDが指定されている場合、検索
        if update.parent_node_id:
            parent_search = state["root_node"].find_node_by_id(update.parent_node_id)
            if parent_search:
                parent_node = parent_search
        
        # 新しいノードを作成して追加
        new_node = ROINode(
            name=update.name,
            details=update.details,
            importance_factor=update.importance_factor,
            value=update.value
        )
        
        parent_node.add_child(new_node)
        
        # ユーザーフィードバック用の更新説明
        update_descriptions.append(
            f"ノード '{new_node.name}' を '{parent_node.name}' の下に追加しました "
            f"（重要度係数: {new_node.importance_factor:.1%}）"
        )
        
        # 現在のフォーカスを新しいノードに更新
        state["current_node_id"] = new_node.node_id
        
        # ノードパスを更新
        if parent_node.node_id in state["node_path"]:
            # パス内の親の位置を検索
            idx = state["node_path"].index(parent_node.node_id)
            # パスを親まで切り詰めて新しいノードを追加
            state["node_path"] = state["node_path"][:idx+1] + [new_node.node_id]
        else:
            # 新しいノードをパスに追加
            state["node_path"].append(new_node.node_id)
    
    # 重要度係数を正規化
    state["root_node"].normalize_importance_factors()
    
    # 現在のノードをまだ探索履歴にない場合は追加
    if state["current_node_id"] not in state["exploration_history"]:
        state["exploration_history"].append(state["current_node_id"])
    
    return state, update_descriptions


# =========================
# グラフノード（エージェントステップ）
# =========================

def deepdive_conversation(state: DeepdiveState) -> DeepdiveState:
    """
    ユーザーの入力に基づいてROIツリー探索を行う（自動会話生成なし）
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
    
    # 関連する過去のリフレクションを取得
    relevant_reflections = reflection_manager.get_relevant_reflections(
        f"{current_node_name} {current_node_details}"
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
ROIツリーへようこそ！現在、「{current_node_name}」ノードに焦点を当てています。
このROIツリーは「コスト削減」と「売上増加」の観点から、ビジネス機会を分析するために使用されます。

具体的な質問や、さらに深掘りしたい領域について教えてください。
        """
        response = AIMessage(content=response_content)
    else:
        # 探索プロンプトを使用して応答を生成
        system_message = SystemMessage(content=DEEPDIVE_SYSTEM_PROMPT + 
            f"\n\n以下の過去のリフレクションを考慮してください:\n{reflection_text}\n\n" +
            "特に重要: すべての応答は必ず日本語で行ってください。")
        
        messages = state["messages"][-5:] + [
            HumanMessage(content=f"""
現在のROIツリーコンテキスト:
{tree_context}

現在のフォーカス: {current_node_name}
説明: {current_node_details}

ユーザーからの質問/入力: {latest_user_message}

ROIツリーについての対話を続けてください。質問に回答するか、ROIツリーのさらなる展開についてアドバイスしてください。

重要度係数について考えることを忘れないでください - 各サブコンポーネントは兄弟コンポーネントと比較してどの程度重要ですか？重要度係数は兄弟間で合計100%になるようにしてください。

過去のリフレクションに基づいた改善点も考慮してください。

必ず日本語で回答してください。
""")
        ]
        
        response = exploration_model.invoke([system_message] + messages)
    
    # 応答をメッセージリストに追加（ユーザーメッセージは既に追加されているはず）
    state["messages"].append(response)
    
    # 応答を解析してノード更新を抽出
    updates = parse_roi_node_update(response)
    
    if updates:
        # ROIツリーに更新を適用
        state, update_descriptions = update_roi_tree(state, updates)
    
    # タスクと結果に対する反省を実行
    context = {
        "node_name": current_node_name,
        "node_details": current_node_details,
        "tree_context": tree_context,
    }
    
    task_reflector.run(
        task=f"ROIツリーノード「{current_node_name}」の探索",
        result=response.content,
        context=context
    )
    
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
    
    # 関連する過去のリフレクションを取得
    relevant_reflections = reflection_manager.get_relevant_reflections("ROIツリー探索の完了判断")
    reflection_context = format_reflections(relevant_reflections)
    
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
        reflection_system_message = SystemMessage(content=f"""あなたはROIツリー分析の完全性を評価する専門家です。
あなたの仕事はROIツリーのブランチが十分に探索されたかどうかを判断することです。

十分に探索されたブランチは以下の特徴を持つべきです:
1. 適切な深さ（通常3〜4レベル）
2. 葉ノードに具体的で測定可能な要素を含む
3. カテゴリの主要コンポーネントをカバーしている
4. 兄弟間で合理的な重要度係数が割り当てられている

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
            
            # リフレクション結果を記録
            task_reflector.run(
                task="ROIツリー探索の完了判断",
                result=f"探索完了度: {analysis.deepdive_completion_percentage}%, 理由: {analysis.reason}",
                context={
                    "tree_stats": tree_stats,
                    "needs_further_exploration": analysis.deepdive_needed,
                    "suggested_focus": analysis.suggested_focus
                }
            )
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
    """探索を続けるかどうかを決定"""
    return "continue" if not state["exploration_complete"] else "__end__"


# =========================
# グラフ構築
# =========================

def build_deepdive_graph() -> Any:  # 循環インポートを避けるためAny型を使用
    """深掘りエージェントグラフを構築してコンパイル"""
    graph = StateGraph(DeepdiveState)
    
    # ノード追加
    graph.add_node("deepdive_conversation", deepdive_conversation)
    graph.add_node("self_reflect_deepdive", self_reflect_deepdive)
    
    # エッジ追加
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
    """深掘り状態を初期化"""
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