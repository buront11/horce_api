"""
プロジェクト内のパラメータを管理するためのモジュール．

A) プログラムを書くときにやること．
  1) デフォルトパラメータを `Parameters` クラス内で定義する．
  2) コマンドライン引数を `common_args` 内で定義する．

B) パラメータを指定して実行するときにやること．
  1) `python config.py` とすると，デフォルトパラメータが `parameters.json` というファイルに書き出される．
  2) パラメータを指定する際は，Parametersクラスを書き換えるのではなく，jsonファイル内の値を書き換えて，
  `python -p parameters.json main.py`
  のようにjsonファイルを指定する．
"""

from dataclasses import dataclass, field
from utils import dump_params
from argparse import ArgumentParser


@dataclass(frozen=True)
class Parameters:
    """
    プログラム全体を通して共通のパラメータを保持するクラス．
    ここにプロジェクト内で使うパラメータを一括管理する．
    """
    ARGS: dict = field(default_factory=lambda: {})  # コマンドライン引数
    RUN_DATE: str = ''  # 実行時の時刻
    GIT_REVISION: str = ''  # 実行時のプログラムのGitのバージョン
    
    # Parameters
    BATCH_SIZE: int = 256
    EPOCHS: int = 30
    LEARNING_RATE: float = 0.001

    HIDDEN_FEAT: int = 2048
    AGGREGATOR_TYPE: str = 'mean'
    POOL_TYPE: str = 'mean'
    GNN_DROPOUT: float = 0.3
    AFFINE_DROPOUT: float = 0.5


def common_args(parser: 'ArgumentParser'):
    """
    コマンドライン引数を定義する関数．
    Args:
        parser (:obj: ArgumentParser):
    """
    parser.add_argument("-p", "--parameters", help="パラメータ設定ファイルのパスを指定．デフォルトはNone", type=str, default=None)
    parser.add_argument("-a", "--arg1", type=int, help="arg1の説明", default=0)  # コマンドライン引数を指定
    parser.add_argument("--arg2", type=float, help="arg2の説明", default=1.0)  # コマンドライン引数を指定
    return parser


if __name__ == "__main__":
    dump_params(Parameters(), './', partial=True)  # デフォルトパラメータを
