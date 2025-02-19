# roi_agents/domain/roitree.py

from typing import Dict, List, Optional

class ROINode:
    """
    シンプルなROIツリーノードの例。
    - name: ノード名 (例: "CostReduction", "CR1: 作業時間削減" など)
    - details: 追加情報(定量目標やメモなど)
    - children: 子ノード(下位階層)
    """
    def __init__(self, name: str, details: Optional[str] = None):
        self.name = name
        self.details = details
        self.children: List["ROINode"] = []

    def add_child(self, child: "ROINode"):
        self.children.append(child)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "details": self.details,
            "children": [c.to_dict() for c in self.children]
        }

def print_roi_tree(node: ROINode, indent: int = 0):
    """ツリーを見やすいテキストでプリントする簡易ヘルパー"""
    prefix = "  " * indent
    print(f"{prefix}- {node.name}")
    if node.details:
        print(f"{prefix}  (details: {node.details})")
    for child in node.children:
        print_roi_tree(child, indent + 1)
