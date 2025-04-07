"""
agents/nlu_agent.py
Natural Language Understanding agent for ROI chat interactions with enhanced ROI extraction
"""
from typing import Dict, List, Optional, Any, Tuple
import json
import re

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class NLUAgent:
    """
    自然言語理解を行い、ユーザーのメッセージからノード情報と数値を抽出するエージェント
    """
    def __init__(self, model_name="gpt-4o"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,
            streaming=False  # 解析には素早いレスポンスが必要
        )
        
        # 解析用のプロンプト
        self.extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。ユーザーの自然言語メッセージからノード名と数値情報を正確に抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_node": true/false,  // ノード名が見つかったかどうか
  "node_name": "抽出したノード名", // 見つかった場合のノード名
  "found_value": true/false,  // 数値が見つかったかどうか
  "value": "抽出した数値",     // 見つかった場合の数値（単位を含む）
  "value_unit": "単位",       // 数値の単位（円、時間、件、人など）
  "value_type": "数値の種類",  // 数値の種類（金額、時間、件数、人数など）
  "confidence": 0-100        // 抽出結果の確信度（0-100）
}}
```

注意:
- 数値は「円」「万円」「億円」「%」「時間」「件」「人」などの単位を含めて抽出してください
- ノード名は完全一致でなくても、明らかに指しているノードがあれば抽出してください
- 抽出できない場合は対応するフィールドをfalseにしてください
- 確信度は抽出結果の信頼性を0-100で表してください
"""),
            ("human", """以下のユーザーメッセージから、ノード名と数値情報を抽出してください。

ユーザーメッセージ: "{message}"

利用可能なノード:
{node_list}

JSON形式で回答してください。
""")
        ])
        
        # 単位変換の分析プロンプト
        self.unit_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。ユーザーの自然言語メッセージから数値情報の単位や型を抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_value": true/false,  // 数値が見つかったかどうか
  "value": 数値,              // 見つかった場合の数値
  "value_unit": "単位",       // 単位（円、時間、件、人など）
  "value_type": "数値の種類",  // 種類（金額、時間、件数、人数など）
  "needs_conversion": true/false, // ROI計算のために変換が必要か
  "target_unit": "目標単位",   // 変換後の目標単位（通常は「円」）
  "additional_info_needed": ["必要な追加情報1", "必要な追加情報2"]
}}
```

注意:
- 数値の単位は「円」「万円」「億円」「%」「時間」「件」「人」などを考慮してください
- 数値型は「金額」「時間」「件数」「人数」「割合」などを考慮してください
- ROI計算のためには最終的に金額（円）への変換が必要です
- 変換に必要な追加情報を特定してください（例: 時給、工数など）
"""),
            ("human", """以下のユーザーメッセージから、数値情報の単位や型を抽出してください。

ユーザーメッセージ: "{message}"

JSON形式で回答してください。
""")
        ])
        
        # 新規: ROI情報抽出プロンプト
        self.roi_extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI分析の専門家です。ユーザーの自然言語メッセージからROI関連情報（コストとベネフィット）を抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_cost": true/false,         // コスト情報が見つかったかどうか
  "cost_value": 数値,               // 見つかった場合のコスト値
  "cost_unit": "単位",              // コストの単位（円、万円、億円など）
  "found_benefit": true/false,      // ベネフィット情報が見つかったかどうか
  "benefit_value": 数値,            // 見つかった場合のベネフィット値
  "benefit_unit": "単位",           // ベネフィットの単位（円、万円、億円など）
  "roi_mentioned": true/false,      // ROIが明示的に言及されているか
  "roi_percentage": 数値,           // 言及されている場合のROI値（%）
  "implementation_period": "期間",  // 実装期間（言及されている場合）
  "priority": 数値,                 // 優先度（1-5、言及されている場合）
  "confidence": 0-100               // 抽出結果の確信度（0-100）
}}
```

注意:
- コストとベネフィットの両方を抽出するよう努めてください
- 単位は「円」「万円」「億円」などを正確に識別してください
- ROIが明示的に言及されている場合はそれも抽出してください
- 実装期間や優先度も言及されていれば抽出してください
- 数値を見つけられない場合は対応するフィールドをfalseにしてください
- 確信度は抽出結果の信頼性を0-100で表してください
"""),
            ("human", """以下のユーザーメッセージから、ROI関連情報を抽出してください。

ユーザーメッセージ: "{message}"

対象ノード: {node_name}

JSON形式で回答してください。
""")
        ])
    
    def extract_node_and_value(self, message: str, mermaid_diagram: str) -> Dict:
        """ユーザーメッセージからノード名と数値を抽出する"""
        # 利用可能なノードのリストを作成
        leaf_nodes = self.extract_leaf_nodes(mermaid_diagram)
        node_list = "\n".join([f"- {label}" for _, label in leaf_nodes])
        
        # LLMで解析
        result = self.llm.invoke(
            self.extract_prompt.format(
                message=message,
                node_list=node_list
            )
        )
        
        # JSONを抽出
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                extraction_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                extraction_result = json.loads(result.content)
            
            # ノードIDの検索（抽出されたノード名に近いノードを検索）
            node_id = None
            if extraction_result.get("found_node", False) and extraction_result.get("node_name"):
                node_name = extraction_result["node_name"]
                
                # ノード名が類似するノードを検索
                for nid, label in leaf_nodes:
                    # 簡易的な類似度チェック（部分文字列）
                    if node_name.lower() in label.lower() or label.lower() in node_name.lower():
                        node_id = nid
                        extraction_result["matched_label"] = label
                        break
            
            extraction_result["node_id"] = node_id
            return extraction_result
            
        except Exception as e:
            print(f"JSON解析エラー: {e}")
            return {
                "found_node": False,
                "found_value": False,
                "confidence": 0,
                "error": str(e)
            }
    
    def analyze_unit_conversion(self, message: str) -> Dict:
        """ユーザーメッセージから数値情報の単位や型を分析し、変換に必要な情報を特定する"""
        # LLMで解析
        result = self.llm.invoke(
            self.unit_prompt.format(
                message=message
            )
        )
        
        # JSONを抽出
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                analysis_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                analysis_result = json.loads(result.content)
            
            return analysis_result
            
        except Exception as e:
            print(f"JSON解析エラー: {e}")
            return {
                "found_value": False,
                "needs_conversion": False,
                "error": str(e)
            }
    
    # 新規: ROI関連情報を抽出するメソッド
    def extract_roi_information(self, message: str, node_name: str = "") -> Dict:
        """
        ユーザーメッセージからROI関連情報（コスト、ベネフィット、ROI等）を抽出する
        
        Args:
            message: ユーザーメッセージ
            node_name: 対象ノード名（オプション）
            
        Returns:
            ROI関連情報を含む辞書
        """
        # LLMで解析
        result = self.llm.invoke(
            self.roi_extract_prompt.format(
                message=message,
                node_name=node_name
            )
        )
        
        # JSONを抽出
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                roi_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                roi_result = json.loads(result.content)
            
            # 単位を考慮して数値を処理
            if roi_result.get("found_cost", False) and roi_result.get("cost_value") is not None:
                cost_unit = roi_result.get("cost_unit", "円")
                cost_value = roi_result["cost_value"]
                
                # 単位に基づいて変換
                if cost_unit == "億円":
                    roi_result["cost_value_yen"] = cost_value * 100000000
                elif cost_unit == "万円":
                    roi_result["cost_value_yen"] = cost_value * 10000
                elif cost_unit == "千円":
                    roi_result["cost_value_yen"] = cost_value * 1000
                else:  # "円"
                    roi_result["cost_value_yen"] = cost_value
            
            if roi_result.get("found_benefit", False) and roi_result.get("benefit_value") is not None:
                benefit_unit = roi_result.get("benefit_unit", "円")
                benefit_value = roi_result["benefit_value"]
                
                # 単位に基づいて変換
                if benefit_unit == "億円":
                    roi_result["benefit_value_yen"] = benefit_value * 100000000
                elif benefit_unit == "万円":
                    roi_result["benefit_value_yen"] = benefit_value * 10000
                elif benefit_unit == "千円":
                    roi_result["benefit_value_yen"] = benefit_value * 1000
                else:  # "円"
                    roi_result["benefit_value_yen"] = benefit_value
            
            # 自動計算されたROIを追加
            if roi_result.get("cost_value_yen") is not None and roi_result.get("benefit_value_yen") is not None:
                cost = roi_result["cost_value_yen"]
                benefit = roi_result["benefit_value_yen"]
                
                if cost > 0:
                    roi_result["calculated_roi"] = (benefit - cost) / cost * 100
                else:
                    roi_result["calculated_roi"] = float('inf') if benefit > 0 else 0
            
            return roi_result
            
        except Exception as e:
            print(f"ROI情報抽出エラー: {e}")
            return {
                "found_cost": False,
                "found_benefit": False,
                "roi_mentioned": False,
                "confidence": 0,
                "error": str(e)
            }
    
    def analyze_conversation(self, conversation_history: List, current_message: str, mermaid_diagram: str) -> Dict:
        """会話の文脈を考慮してメッセージを分析する"""
        # まず単純に現在のメッセージだけで分析
        initial_result = self.extract_node_and_value(current_message, mermaid_diagram)
        
        # 高確信度の結果が得られた場合はそのまま返す
        if initial_result.get("confidence", 0) > 80:
            return initial_result
        
        # 低確信度の場合は、会話履歴も含めて再分析
        recent_context = "\n".join([
            f"{'ユーザー' if role == 'user' else 'アシスタント'}: {content}"
            for role, content in conversation_history[-3:] if role in ['user', 'assistant']
        ])
        
        context_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROIツリー分析の専門家です。会話の文脈を考慮して、最新のユーザーメッセージからノード名と数値情報を抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_node": true/false,
  "node_name": "抽出したノード名",
  "found_value": true/false,
  "value": "抽出した数値",
  "value_unit": "単位",
  "value_type": "数値の種類",
  "confidence": 0-100
}}
```"""),
            ("human", """以下の会話の文脈を考慮して、最新のユーザーメッセージからノード名と数値情報を抽出してください。

会話の文脈:
{{context}}

最新のメッセージ: "{{message}}"

利用可能なノード:
{[node_list}}

JSON形式で回答してください。
""")
        ])
        
        # 利用可能なノードのリストを作成
        leaf_nodes = self.extract_leaf_nodes(mermaid_diagram)
        node_list = "\n".join([f"- {label}" for _, label in leaf_nodes])
        
        result = self.llm.invoke(
            context_prompt.format(
                context=recent_context,
                message=current_message,
                node_list=node_list
            )
        )
        
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                context_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                context_result = json.loads(result.content)
            
            # ノードIDの検索
            node_id = None
            if context_result.get("found_node", False) and context_result.get("node_name"):
                node_name = context_result["node_name"]
                
                for nid, label in leaf_nodes:
                    if node_name.lower() in label.lower() or label.lower() in node_name.lower():
                        node_id = nid
                        context_result["matched_label"] = label
                        break
            
            context_result["node_id"] = node_id
            return context_result
            
        except Exception as e:
            print(f"文脈解析エラー: {e}")
            return initial_result  # エラーの場合は最初の結果を返す
    
    # 新規: 会話からROI情報を抽出するメソッド
    def analyze_roi_from_conversation(self, conversation_history: List, current_message: str, node_name: str = "") -> Dict:
        """
        会話の文脈を考慮してROI関連情報を抽出する
        
        Args:
            conversation_history: 会話履歴
            current_message: 現在のユーザーメッセージ
            node_name: 対象ノード名（オプション）
            
        Returns:
            ROI関連情報を含む辞書
        """
        # まず単純に現在のメッセージだけで分析
        initial_result = self.extract_roi_information(current_message, node_name)
        
        # 高確信度の結果が得られた場合はそのまま返す
        if initial_result.get("confidence", 0) > 80:
            return initial_result
        
        # 低確信度の場合は、会話履歴も含めて再分析
        recent_context = "\n".join([
            f"{'ユーザー' if role == 'user' else 'アシスタント'}: {content}"
            for role, content in conversation_history[-3:] if role in ['user', 'assistant']
        ])
        
        context_roi_prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはROI分析の専門家です。会話の文脈を考慮して、最新のユーザーメッセージからROI関連情報を抽出してください。

以下のJSON形式で回答してください:
```json
{{
  "found_cost": true/false,
  "cost_value": 数値,
  "cost_unit": "単位",
  "found_benefit": true/false,
  "benefit_value": 数値,
  "benefit_unit": "単位",
  "roi_mentioned": true/false,
  "roi_percentage": 数値,
  "implementation_period": "期間",
  "priority": 数値,
  "confidence": 0-100
}}
```"""),
            ("human", """以下の会話の文脈を考慮して、最新のユーザーメッセージからROI関連情報を抽出してください。

会話の文脈:
{{context}}

最新のメッセージ: "{{message}}"

対象ノード: {{node_name}}

JSON形式で回答してください。
""")
        ])
        
        result = self.llm.invoke(
            context_roi_prompt.format(
                context=recent_context,
                message=current_message,
                node_name=node_name
            )
        )
        
        try:
            # 応答からJSON部分を抽出
            pattern = r"```json\n(.*?)\n```"
            matches = re.search(pattern, result.content, re.DOTALL)
            
            if matches:
                json_str = matches.group(1)
                context_result = json.loads(json_str)
            else:
                # JSONブロックがない場合、全体を解析
                context_result = json.loads(result.content)
            
            # 単位を考慮して数値を処理
            if context_result.get("found_cost", False) and context_result.get("cost_value") is not None:
                cost_unit = context_result.get("cost_unit", "円")
                cost_value = context_result["cost_value"]
                
                # 単位に基づいて変換
                if cost_unit == "億円":
                    context_result["cost_value_yen"] = cost_value * 100000000
                elif cost_unit == "万円":
                    context_result["cost_value_yen"] = cost_value * 10000
                elif cost_unit == "千円":
                    context_result["cost_value_yen"] = cost_value * 1000
                else:  # "円"
                    context_result["cost_value_yen"] = cost_value
            
            if context_result.get("found_benefit", False) and context_result.get("benefit_value") is not None:
                benefit_unit = context_result.get("benefit_unit", "円")
                benefit_value = context_result["benefit_value"]
                
                # 単位に基づいて変換
                if benefit_unit == "億円":
                    context_result["benefit_value_yen"] = benefit_value * 100000000
                elif benefit_unit == "万円":
                    context_result["benefit_value_yen"] = benefit_value * 10000
                elif benefit_unit == "千円":
                    context_result["benefit_value_yen"] = benefit_value * 1000
                else:  # "円"
                    context_result["benefit_value_yen"] = benefit_value
            
            # 自動計算されたROIを追加
            if context_result.get("cost_value_yen") is not None and context_result.get("benefit_value_yen") is not None:
                cost = context_result["cost_value_yen"]
                benefit = context_result["benefit_value_yen"]
                
                if cost > 0:
                    context_result["calculated_roi"] = (benefit - cost) / cost * 100
                else:
                    context_result["calculated_roi"] = float('inf') if benefit > 0 else 0
            
            return context_result
            
        except Exception as e:
            print(f"ROI文脈解析エラー: {e}")
            return initial_result  # エラーの場合は最初の結果を返す
    
    def extract_leaf_nodes(self, mermaid_code):
        """
        Mermaidコードから末端ノード（他のノードの親になっていないノード）を抽出する
        """
        all_node_ids = set()
        parent_node_ids = set()
        node_definitions = {}
        lines = mermaid_code.strip().split('\n')
        
        # まずノードの定義をキャプチャ
        for line in lines:
            # ノード定義をキャプチャ（例: node1["テキスト"]）
            matches = re.findall(r'(\w+)\[\"([^\"]+)\"', line)
            for match in matches:
                node_id = match[0]
                node_label = match[1]
                all_node_ids.add(node_id)
                node_definitions[node_id] = node_label
        
        # 次に親ノードを見つける
        for line in lines:
            # エッジ定義をキャプチャ（例: node1 --> node2）
            edge_matches = re.findall(r'(\w+)\s*-->', line)
            for match in edge_matches:
                parent_node_ids.add(match)
        
        # 末端ノードは、親になっていないノード
        leaf_nodes = all_node_ids - parent_node_ids
        
        # 末端ノードとそのラベルを返す
        result = [(node_id, node_definitions.get(node_id, "")) for node_id in leaf_nodes]
        return result
    
    # 新規: ノードからコストとベネフィットを抽出する
    def extract_cost_benefit_from_node_text(self, node_text: str) -> Tuple[Optional[float], Optional[float]]:
        """
        ノードテキストからコストとベネフィットを抽出する
        
        Args:
            node_text: ノードテキスト
            
        Returns:
            (コスト, ベネフィット)のタプル。抽出できない場合はNone
        """
        # 数値パターン（コスト、ベネフィット、ROI）
        cost_pattern = r'コスト[:：]?\s*(\d+[,.]?\d*)\s*([億万千]?円)'
        benefit_pattern = r'(効果|ベネフィット|利益)[:：]?\s*(\d+[,.]?\d*)\s*([億万千]?円)'
        roi_pattern = r'ROI[:：]?\s*(\d+[,.]?\d*)\s*%'
        
        cost_value = None
        cost_matches = re.search(cost_pattern, node_text)
        if cost_matches:
            try:
                value = float(cost_matches.group(1).replace(',', ''))
                unit = cost_matches.group(2)
                
                # 単位に基づいて変換
                if unit == '億円':
                    cost_value = value * 100000000
                elif unit == '万円':
                    cost_value = value * 10000
                elif unit == '千円':
                    cost_value = value * 1000
                else:  # '円'
                    cost_value = value
            except ValueError:
                pass
        
        benefit_value = None
        benefit_matches = re.search(benefit_pattern, node_text)
        if benefit_matches:
            try:
                value = float(benefit_matches.group(2).replace(',', ''))
                unit = benefit_matches.group(3)
                
                # 単位に基づいて変換
                if unit == '億円':
                    benefit_value = value * 100000000
                elif unit == '万円':
                    benefit_value = value * 10000
                elif unit == '千円':
                    benefit_value = value * 1000
                else:  # '円'
                    benefit_value = value
            except ValueError:
                pass
        
        # ROIからの逆算（コストがあり、ベネフィットがない場合）
        if cost_value is not None and benefit_value is None:
            roi_matches = re.search(roi_pattern, node_text)
            if roi_matches:
                try:
                    roi_percentage = float(roi_matches.group(1))
                    # ROI = (ベネフィット - コスト) / コスト × 100%
                    # ベネフィット = (ROI × コスト / 100) + コスト
                    benefit_value = (roi_percentage * cost_value / 100) + cost_value
                except ValueError:
                    pass
        
        return cost_value, benefit_value