# roi_agents/agents/proposal_agent.py

import os
from typing import Literal

# langgraph関連
from langgraph.graph import StateGraph, END

# langchain関連
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from domain.schemas import ProposalState
from domain.roitree import ROINode
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model_name="gpt-4o", 
    temperature=0.5,
    streaming=False,
)

def node_tree_to_string(node: ROINode, indent:int=0) -> str:
    prefix = "  " * indent
    lines = [f"{prefix}- {node.name}"]
    if node.details:
        lines.append(f"{prefix}  (details: {node.details})")
    for child in node.children:
        lines.append(node_tree_to_string(child, indent+1))
    return "\n".join(lines)

def propose_solutions(state: ProposalState) -> ProposalState:
    """
    既存のROIツリーを元に、提案を行う会話を進める。
    ノードごとに質問して情報を集め、ROIを試算する想定。
    """
    system_prompt = SystemMessage(content="""
あなたはROI試算アシスタントです。
与えられたROIツリーの各ノードについて、必要な追加情報をユーザに尋ね、ROIの見積もりを行ってください。
最終的にROIが確定したらタスクを終了してください。
""")

    node_list_str = node_tree_to_string(state["root_node"])
    user_prompt = f"""\
ROIツリー:
{node_list_str}

このツリーをもとに提案や試算を進めてください。
あなたは「提案エージェント」です。ROI試算に必要な質問を行い、最終的な結論を出してください。
"""

    ai_message = model([system_prompt, *state["messages"], HumanMessage(content=user_prompt)])
    state["messages"].append(HumanMessage(content=user_prompt))
    state["messages"].append(ai_message)

    # ダミーでROI算出結果を入れてみる
    state["roi_calculations"]["summary"] = "仮のROI試算結果: コスト削減500万円、売上増1,000万円"

    return state

def self_reflect_proposal(state: ProposalState) -> ProposalState:
    """
    ある程度の試算ができたら終了とする簡易例。
    iteration_countで終了を判定。
    """
    state["iteration_count"] += 1
    if state["iteration_count"] >= state["max_iterations"]:
        state["proposal_complete"] = True
    else:
        state["proposal_complete"] = False
    return state

def should_continue_proposal(state: ProposalState) -> Literal["continue", "__end__"]:
    return "continue" if not state["proposal_complete"] else "__end__"

def build_proposal_graph():
    graph = StateGraph(ProposalState)

    graph.add_node("propose_solutions", propose_solutions)
    graph.add_node("self_reflect_proposal", self_reflect_proposal)

    graph.add_edge("propose_solutions", "self_reflect_proposal")
    graph.add_conditional_edges(
        "self_reflect_proposal",
        should_continue_proposal,
        {"continue": "propose_solutions", "__end__": END}
    )

    graph.set_entry_point("propose_solutions")
    return graph.compile()
