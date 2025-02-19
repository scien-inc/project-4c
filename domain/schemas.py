# roi_agents/domain/schemas.py

from typing import TypedDict, List, Dict, Any, Literal, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage  # langchain_coreでなく langchain.schema を想定
from domain.roitree import ROINode

# =========================
# 課題エージェント用のステート
# =========================
class DeepdiveState(TypedDict):
    # 対話履歴（シンプルにBaseMessage継承クラスを格納）
    messages: List[BaseMessage]

    # ROIノードのルート（例えば "Gain" ノードを頂点に持つツリーを想定）
    root_node: Optional[ROINode]

    # 現在のノード: ユーザと対話中のノード(どこを深掘りしているか)
    current_node: Optional[ROINode]

    # フラグ類
    iteration_count: int
    max_iterations: int
    # 「まだ深掘りが必要か」or「十分に深掘りできたか」をセルフリフレクションで判断
    deepdive_needed: bool


# =========================
# 提案エージェント用のステート
# =========================
class ProposalState(TypedDict):
    # 対話履歴
    messages: List[BaseMessage]

    # 課題側で構築されたROIツリーを参照（read only）
    root_node: ROINode

    # 提案内容や試算結果を保持
    roi_calculations: Dict[str, Any]

    # フラグ類
    iteration_count: int
    max_iterations: int
    # 「ROIが最終的に算出できたか？」セルフリフレクションで判断
    proposal_complete: bool
