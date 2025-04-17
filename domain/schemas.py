"""
domain/schemas.py
Enhanced schema definitions for ROI analysis with granularity assessment and DX tool support
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NumericalValue(BaseModel):
    """数値情報を表すクラス"""
    value: float
    unit: str = ""
    description: str = ""


class UnitConversion(BaseModel):
    """単位変換情報を表す構造体"""
    original_value: float
    original_unit: str
    converted_value: float
    converted_unit: str
    conversion_factor: float
    calculation: str
    explanation: str


class NodeGranularityIssue(BaseModel):
    """ノードの粒度の問題を表す構造体"""
    name: str
    issue: str
    suggestion: str
    dx_tools: List[str] = Field(default_factory=list)


class LeafNodeAnalysis(BaseModel):
    """
    末端ノード分析結果を表すクラス
    欠けているフィールドに関する問題を解決するためにデフォルト値を設定
    """
    # エラーで要求されている必須フィールド
    has_numerical_data: bool = Field(default=False)
    missing_nodes: List[str] = Field(default_factory=list)
    incomplete_nodes: List[Dict[str, str]] = Field(default_factory=list)
    completion_percentage: float = Field(default=0.0)
    
    # 元の実装からのフィールド
    granularity_issues: List[Dict[str, Any]] = Field(default_factory=list)
    branching_issues: List[Dict[str, Any]] = Field(default_factory=list)


class SolutionRecommendation(BaseModel):
    """ソリューション推薦を表す構造体（DXツール対応）"""
    leaf_node_id: str
    leaf_node_name: str
    solution_name: str
    solution_description: str
    dx_tools: List[str] = Field(default_factory=list)
    estimated_cost: Dict[str, Any]
    estimated_benefit: Dict[str, Any]
    implementation_timeframe: str
    priority: int
    common_parameters: List[str] = Field(default_factory=list)
    node_specific_parameters: List[str] = Field(default_factory=list)


class ProposalResult(BaseModel):
    """提案結果の概要を表す構造体（簡素化ROI計算対応）"""
    total_investment: float
    total_benefit: float
    roi_percentage: float
    implementation_timeframe: str
    key_recommendations: List[str]
    priority_ranking: Optional[List[Dict[str, Any]]] = None
    
    # 元の要約情報（特に単一ノード提案用）
    original_summary: Optional[Dict[str, Any]] = None
    # 対象ノード（単一ノード提案用）
    target_node: Optional[str] = None
    # 単年ROIデータ
    single_year_roi: Optional[Dict[str, Any]] = None


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


class DXToolRecommendation(BaseModel):
    """DXツール推薦を表す構造体"""
    name: str
    description: str
    estimated_cost: Dict[str, Any] = Field(default_factory=dict)
    implementation_complexity: int = 3  # 1-5の複雑さ（1=簡単、5=複雑）
    required_skills: List[str] = Field(default_factory=list)
    integration_points: List[str] = Field(default_factory=list)
    typical_roi: Optional[float] = None


class ParameterCollection(BaseModel):
    """パラメータ収集を表す構造体"""
    common_parameters: List[str] = Field(default_factory=list)
    node_specific_parameters: List[str] = Field(default_factory=list)
    collected_parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """全てのパラメータが収集されたかを確認する"""
        all_params = self.common_parameters + self.node_specific_parameters
        return len(self.collected_parameters) >= len(all_params)
    
    def get_missing_parameters(self) -> List[str]:
        """未収集のパラメータを取得する"""
        all_params = self.common_parameters + self.node_specific_parameters
        collected_params = self.collected_parameters.keys()
        return [p for p in all_params if p not in collected_params]


class ROICalculation(BaseModel):
    """ROI計算結果を表す構造体（単年ROI中心）"""
    investment: Dict[str, float]
    benefit: Dict[str, float]
    roi: Dict[str, float]
    explanation: str
    assumptions: List[str] = Field(default_factory=list)
    
    def get_first_year_roi(self) -> float:
        """初年度ROIを取得する"""
        return self.roi.get("first_year", 0.0)
    
    def get_payback_period(self) -> float:
        """投資回収期間を取得する"""
        return self.roi.get("payback_period", 0.0)