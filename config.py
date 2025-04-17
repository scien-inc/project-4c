"""
config.py
環境変数と設定の管理
"""
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# APIキーの取得
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# デバッグ情報
def debug_api_key():
    """APIキーのデバッグ情報を出力する"""
    if OPENAI_API_KEY:
        # APIキーの最初と最後の数文字だけを表示して安全にデバッグ
        print(f"APIキー設定状況: {OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:]}")
        return True
    else:
        print("APIキーが設定されていません")
        return False

# デフォルトのモデル設定
DEFAULT_MODEL = "o3-mini"

# 他の設定
MAX_TOKENS = 4000
#DEFAULT_TEMPERATURE = 0.2