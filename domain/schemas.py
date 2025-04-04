"""
domain/schemas.py
Schema definitions for ROI analysis
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NumericalValue(BaseModel):
    """数値情報を表す構造体"""
    value: float
    unit: str
    type: str


class UnitConversion(BaseModel):
    """単位変換情報を表す構造体"""
    original_value: float
    original_unit: str
    converted_value: float
    converted_unit: str
    conversion_factor: float
    calculation: str
    explanation: str


class LeafNodeAnalysis(BaseModel):
    """末端ノード分析結果を表す構造体"""
    has_numerical_data: bool
    missing_nodes: List[str]
    incomplete_nodes: List[Dict[str, str]]
    completion_percentage: float


class SolutionRecommendation(BaseModel):
    """ソリューション推薦を表す構造体"""
    leaf_node_id: str
    leaf_node_name: str
    solution_name: str
    solution_description: str
    estimated_cost: Dict[str, Any]
    estimated_benefit: Dict[str, Any]
    implementation_timeframe: str
    priority: int


class ProposalResult(BaseModel):
    """提案結果の概要を表す構造体"""
    total_investment: float
    total_benefit: float
    roi_percentage: float
    implementation_timeframe: str
    key_recommendations: List[str]
    priority_ranking: Optional[List[Dict[str, Any]]] = None
    
    # 新規: 元の要約情報（特に単一ノード提案用）
    original_summary: Optional[Dict[str, Any]] = None
    # 新規: 対象ノード（単一ノード提案用）
    target_node: Optional[str] = None


# 新規: ROI情報を表す構造体
class ROIData(BaseModel):
    """ROI情報を表す構造体"""
    node_id: str
    node_name: str
    cost: float
    cost_unit: str = "円"
    benefit: float
    benefit_unit: str = "円"
    roi_percentage: float
    implementation_period: Optional[str] = None
    priority: Optional[int] = None
    confidence: Optional[float] = None  # 推定値の信頼度（0-100）
    
    def get_formatted_roi(self) -> str:
        """フォーマットされたROI文字列を取得する"""
        return f"{self.roi_percentage:.1f}%"
    
    def get_formatted_cost(self) -> str:
        """フォーマットされたコスト文字列を取得する"""
        if self.cost >= 100000000:  # 1億円以上
            return f"{self.cost/100000000:.1f}億円"
        elif self.cost >= 10000:  # 1万円以上
            return f"{self.cost/10000:.1f}万円"
        else:
            return f"{self.cost:.0f}円"
    
    def get_formatted_benefit(self) -> str:
        """フォーマットされたベネフィット文字列を取得する"""
        if self.benefit >= 100000000:  # 1億円以上
            return f"{self.benefit/100000000:.1f}億円"
        elif self.benefit >= 10000:  # 1万円以上
            return f"{self.benefit/10000:.1f}万円"
        else:
            return f"{self.benefit:.0f}円"


# 新規: ノードのリスク-リターン情報を表す構造体
class NodeRiskReturn(BaseModel):
    """ノードのリスク-リターン情報を表す構造体"""
    node_id: str
    node_name: str
    risk_level: float  # 0-100のスケール
    return_level: float  # 0-100のスケール
    implementation_difficulty: float  # 0-100のスケール
    strategic_importance: float  # 0-100のスケール
    roi_percentage: float
    estimated_timeframe: str
    
    def get_risk_return_ratio(self) -> float:
        """リスク-リターン比率を計算する"""
        if self.risk_level == 0:
            return float('inf')  # リスクがゼロの場合は無限大
        return self.return_level / self.risk_level