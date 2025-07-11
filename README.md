# ウェーブレットとL1畳み込みスパース表現を用いた単一画像解像度

この研究では、畳み込みスパース表現 (CSR) の誤差項の L2 ノルムを L1 ノルムに変更しただけです。研究の手順はすべて、比較研究 [1] で著者が行ったものと同じです。

ただし、この研究と比較研究の間にはいくつかの違いがあります：
1) 比較研究では、著者は RGB チャネルを考慮して画像を処理しましたが、この研究ではグレースケール画像のみを処理しました (メモリが十分でなかったため)。
2) この研究では、辞書の数が少なく、パラメータも小さくしました (メモリが十分でなかったため)。
3) この研究では、マッピング関数のトレーニングが比較研究とかなり異なっていました (比較研究の論文の説明が鮮明すぎて、私には理解しにくいです)。

この研究を改善するための提案がいくつかあります：
1) メモリの問題に対処するために、研究室に大きなメインメモリサーバーを 1 つ作成してください。または、最初から高専のメモリサーバーを使用してください。
2) マッピング関数のトレーニングを完全に理解し、適切に行うために、比較研究の論文を確認してください。
3) PSNR の結果に実際に問題があったため、私のコードを確認してください。正しい結果は、研究室で印刷された論文ではなく、チームにアップロードした論文に記載されているはずです。

参考文献：
[1] Convolutional Sparse Coding Using Wavelets for Single Image Super-Resolution
    https://ieeexplore.ieee.org/document/8807113
