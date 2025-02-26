"""
リフレクション管理システム - ROIツリー探索および提案プロセスのための自己反省機能
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class JudgmentResult(str, Enum):
    """リフレクション判断の結果を表す列挙型"""
    SUFFICIENT = "sufficient"  # 十分な結果が得られた
    INSUFFICIENT = "insufficient"  # 不十分な結果（再試行が必要）
    NEEDS_REFINEMENT = "needs_refinement"  # 改善が必要（別のアプローチが必要）


class ReflectionJudgment(BaseModel):
    """リフレクションの判断結果"""
    result: JudgmentResult = Field(..., description="判断結果")
    reason: str = Field(..., description="判断の理由")
    needs_retry: bool = Field(..., description="再試行が必要かどうか")
    suggested_approach: Optional[str] = Field(None, description="推奨されるアプローチ（必要な場合）")


class Reflection(BaseModel):
    """リフレクションのデータモデル"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="リフレクションの一意のID")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="リフレクションのタイムスタンプ")
    task: str = Field(..., description="実行されたタスクまたはノード")
    context: Dict[str, Any] = Field(default_factory=dict, description="タスク実行時のコンテキスト情報")
    result: str = Field(..., description="タスク実行の結果")
    reflection: str = Field(..., description="タスクと結果に対するリフレクション")
    judgment: ReflectionJudgment = Field(..., description="リフレクションの判断")
    tags: List[str] = Field(default_factory=list, description="リフレクションのタグ（検索用）")


class ReflectionManager:
    """リフレクションの保存・取得を管理するクラス"""
    
    def __init__(self, file_path: str = "roi_agents/data/reflections.json"):
        """
        ReflectionManagerの初期化
        
        Args:
            file_path: リフレクションを保存するJSONファイルのパス
        """
        self.file_path = file_path
        self.reflections = self._load_reflections()
        
    def _load_reflections(self) -> Dict[str, Reflection]:
        """
        ファイルからリフレクションを読み込む
        
        Returns:
            ID->Reflectionのマッピング
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {id: Reflection(**refl) for id, refl in data.items()}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def _save_reflections(self) -> None:
        """リフレクションをファイルに保存する"""
        try:
            # ディレクトリの作成（存在しない場合）
            import os
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(
                    {id: refl.model_dump() for id, refl in self.reflections.items()},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
        except Exception as e:
            print(f"リフレクションの保存中にエラーが発生しました: {str(e)}")
    
    def add_reflection(self, reflection: Reflection) -> str:
        """
        新しいリフレクションを追加する
        
        Args:
            reflection: 追加するリフレクション
            
        Returns:
            追加されたリフレクションのID
        """
        self.reflections[reflection.id] = reflection
        self._save_reflections()
        return reflection.id
    
    def get_reflection(self, reflection_id: str) -> Optional[Reflection]:
        """
        IDでリフレクションを取得する
        
        Args:
            reflection_id: リフレクションのID
            
        Returns:
            対応するリフレクション、または存在しない場合はNone
        """
        return self.reflections.get(reflection_id)
    
    def get_all_reflections(self) -> List[Reflection]:
        """
        すべてのリフレクションを取得する
        
        Returns:
            すべてのリフレクションのリスト
        """
        return list(self.reflections.values())
    
    def get_relevant_reflections(self, query: str, max_results: int = 5) -> List[Reflection]:
        """
        クエリに関連するリフレクションを取得する
        
        Args:
            query: 検索クエリ
            max_results: 返す最大結果数
            
        Returns:
            関連するリフレクションのリスト
        """
        if not self.reflections:
            return []
        
        # シンプルなキーワードマッチング（実際の実装ではより高度な類似性検索を使用）
        relevant = []
        query_lower = query.lower()
        
        # タスク、リフレクション、タグに基づいて関連性をチェック
        for refl in self.reflections.values():
            score = 0
            
            # タスクにキーワードが含まれるか
            if any(kw in refl.task.lower() for kw in query_lower.split()):
                score += 10
                
            # リフレクションにキーワードが含まれるか
            if any(kw in refl.reflection.lower() for kw in query_lower.split()):
                score += 5
                
            # タグにキーワードが含まれるか
            for tag in refl.tags:
                if any(kw in tag.lower() for kw in query_lower.split()):
                    score += 3
            
            if score > 0:
                relevant.append((refl, score))
        
        # スコアでソートして上位のみ返す
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in relevant[:max_results]]


class TaskReflector:
    """タスク実行の結果について反省するクラス"""
    
    def __init__(self, llm: ChatOpenAI, reflection_manager: ReflectionManager):
        """
        TaskReflectorの初期化
        
        Args:
            llm: 使用する言語モデル
            reflection_manager: リフレクション管理インスタンス
        """
        self.llm = llm
        self.reflection_manager = reflection_manager
        
    def run(self, task: str, result: str, context: Dict[str, Any] = None) -> Reflection:
        """
        タスクとその結果についてリフレクションを行う
        
        Args:
            task: 実行されたタスク
            result: 実行結果
            context: タスクのコンテキスト情報
            
        Returns:
            リフレクション結果
        """
        context = context or {}
        
        # リフレクションプロンプトを作成
        reflection_prompt = ChatPromptTemplate.from_template(
            """あなたはROIツリーの探索と分析プロセスを改善するための専門家です。
タスクとその結果について反省し、プロセスを改善するための洞察を提供してください。

# タスク
{task}

# 実行結果
{result}

# 反省ガイドライン
1. タスクは効果的に実行されましたか？
2. 結果は十分に詳細で具体的ですか？
3. どのような改善点がありますか？
4. このアプローチは効率的でしたか？
5. 同様の問題への今後のアプローチをどのように改善できますか？

あなたの反省を以下の点について詳細に記述してください:
- 何がうまくいったか
- 何が課題だったか
- 今後の改善策

最後に、以下の形式で判断を下してください:
```json
{{
  "result": ["sufficient", "insufficient", "needs_refinement"] のいずれか,
  "reason": "判断の詳細な理由",
  "needs_retry": true/false,
  "suggested_approach": "（必要な場合）推奨されるアプローチ"
}}
```
"""
        )
        
        # リフレクションを実行
        reflection_chain = reflection_prompt | self.llm | StrOutputParser()
        reflection_result = reflection_chain.invoke({
            "task": task,
            "result": result
        })
        
        # 判断部分を抽出
        import re
        judgment_match = re.search(r"```json\s*(.*?)\s*```", reflection_result, re.DOTALL)
        
        if judgment_match:
            try:
                judgment_dict = json.loads(judgment_match.group(1))
                judgment = ReflectionJudgment(**judgment_dict)
            except Exception:
                # デフォルトの判断（リトライなし）
                judgment = ReflectionJudgment(
                    result=JudgmentResult.SUFFICIENT,
                    reason="判断の解析に失敗しました。デフォルトの評価を使用します。",
                    needs_retry=False
                )
        else:
            # デフォルトの判断（リトライなし）
            judgment = ReflectionJudgment(
                result=JudgmentResult.SUFFICIENT,
                reason="判断が見つかりませんでした。デフォルトの評価を使用します。",
                needs_retry=False
            )
        
        # 反省を作成
        reflection = Reflection(
            task=task,
            context=context,
            result=result,
            reflection=reflection_result,
            judgment=judgment,
            tags=self._extract_tags(task, reflection_result)
        )
        
        # リフレクション管理システムに追加
        self.reflection_manager.add_reflection(reflection)
        
        return reflection
    
    def _extract_tags(self, task: str, reflection: str) -> List[str]:
        """
        タスクとリフレクションからタグを抽出する
        
        Args:
            task: タスク
            reflection: リフレクション
            
        Returns:
            抽出されたタグのリスト
        """
        # 簡易的なタグ抽出（実際の実装ではより高度な方法を使用）
        important_keywords = [
            "コスト削減", "売上増加", "ROI", "重要度", "深掘り", "提案", 
            "見積もり", "計算", "分析", "不足", "完了", "継続"
        ]
        
        tags = []
        
        # タスクとリフレクションから重要なキーワードを抽出
        combined_text = (task + " " + reflection).lower()
        for keyword in important_keywords:
            if keyword.lower() in combined_text:
                tags.append(keyword)
        
        return tags


def format_reflections(reflections: List[Reflection]) -> str:
    """
    リフレクションを整形する
    
    Args:
        reflections: 整形するリフレクションのリスト
        
    Returns:
        整形されたリフレクション文字列
    """
    if not reflections:
        return "関連する過去のリフレクションはありません。"
    
    formatted = []
    for i, refl in enumerate(reflections):
        formatted.append(
            f"<リフレクション_{i+1}>\n"
            f"<タスク>{refl.task}</タスク>\n"
            f"<反省>{refl.reflection}</反省>\n"
            f"<判断>{refl.judgment.result}: {refl.judgment.reason}</判断>\n"
            f"</リフレクション_{i+1}>"
        )
    
    return "\n\n".join(formatted)