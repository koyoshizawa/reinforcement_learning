# reinforcement_learning

## 目的
強化学習をいろいろお試しするためのリポジトリ。  
OpenAi Gym を使用して環境を構築。  
最終的にはFxトレードを学習させたい。(野望)  

## フロー
1. cart-pole-v0を試す  
actionはランダム -> Q-learning 予定
2. fx の自作env作成
action -> [[買い, 売り, 保持], 枚数]
observation -> [position, 枚数, time_series_data]
time_series_data -> 1分, 5分, 15分, 30分, 60分 のcloseデータ

## 参考
・DQNをKerasとTensorFlowとOpenAI Gymで実装する  
<http://elix-tech.github.io/ja/2016/06/29/dqn-ja.html>
