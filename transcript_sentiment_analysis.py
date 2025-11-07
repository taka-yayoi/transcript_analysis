# Databricks notebook source
# MAGIC %md
# MAGIC # 文字起こし感情分析パイプライン
# MAGIC
# MAGIC 文字起こしファイルから会話の感情を分析し、発話者間の影響関係を可視化します。
# MAGIC
# MAGIC ## 分析の流れ
# MAGIC 1. **文字起こしファイルの読み込み** - タイムスタンプ付き文字起こしを解析
# MAGIC 2. **感情分析（並列処理）** - Claude Sonnet 4.5で各発話の感情をSpark UDFで並列分析
# MAGIC 3. **発話者別可視化** - 時系列推移、感情分布、スコア分布を発話者別に表示
# MAGIC 4. **感情変動分析** - 急激な感情変化とその原因となった発話を特定
# MAGIC 5. **影響力分析** - どの発話者が他者の感情に最も影響を与えているか分析
# MAGIC
# MAGIC ## 主な機能
# MAGIC - ✅ タイムスタンプ付き文字起こしの自動パース
# MAGIC - ✅ Claude Sonnet 4.5による高精度な感情分析
# MAGIC - ✅ Spark UDFによる並列処理で高速化
# MAGIC - ✅ Plotlyによるインタラクティブな可視化
# MAGIC - ✅ 発話者間の感情的影響関係の特定

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. セットアップ

# COMMAND ----------

# MAGIC %pip install mlflow pandas plotly

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional

from mlflow.deployments import get_deploy_client
import mlflow

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# クライアント初期化
fm_client = get_deploy_client("databricks")

# 出力先
OUTPUT_VOLUME = "/Volumes/takaakiyayoi_catalog/movie_analysis/movie_data"
dbutils.fs.mkdirs(OUTPUT_VOLUME)

print("✅ セットアップ完了")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. 文字起こしファイルの指定
# MAGIC
# MAGIC ### ファイルフォーマット
# MAGIC 以下の形式の文字起こしファイルに対応しています：
# MAGIC ```
# MAGIC [発話者名] HH:MM:SS
# MAGIC 発話内容テキスト
# MAGIC
# MAGIC [発話者名] HH:MM:SS
# MAGIC 発話内容テキスト
# MAGIC ```
# MAGIC
# MAGIC ### アップロード方法:
# MAGIC 1. Databricks UIで Catalog → movie_analysis → movie_data を開く
# MAGIC 2. 「Upload Files」をクリック
# MAGIC 3. 文字起こしファイル（例: transcript_sample.txt）をアップロード
# MAGIC 4. 下のウィジェットにファイル名を入力

# COMMAND ----------

dbutils.widgets.text("transcript_filename", "transcript_sample.txt", "文字起こしファイル名")

# COMMAND ----------

transcript_filename = dbutils.widgets.get("transcript_filename")
transcript_path = f"{OUTPUT_VOLUME}/{transcript_filename}"

print(f"🎯 分析対象: {transcript_path}")

# ファイルの存在確認
try:
    dbutils.fs.ls(transcript_path)
    print(f"✅ ファイル確認完了")
except Exception as e:
    print(f"❌ ファイルが見つかりません: {transcript_path}")
    print(f"   エラー: {e}")
    print(f"\n📁 {OUTPUT_VOLUME} の内容:")
    display(dbutils.fs.ls(OUTPUT_VOLUME))
    raise Exception(f"文字起こしファイルが見つかりません: {transcript_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. 文字起こしファイルの読み込み
# MAGIC
# MAGIC タイムスタンプと発話者情報を含む文字起こしファイルを解析します。
# MAGIC - 各発話の開始時刻と終了時刻を自動計算
# MAGIC - 発話者名を抽出
# MAGIC - セグメント単位でデータフレームに変換

# COMMAND ----------

def load_transcript_from_file(transcript_path: str) -> List[Dict]:
    """
    文字起こしファイルを読み込んでセグメントに分割

    フォーマット想定:
    [名前] HH:MM:SS
    テキスト内容
    """
    print(f"📄 文字起こしファイル読み込み: {transcript_path}")

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        segments = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # [名前] HH:MM:SS 形式をパース
            timestamp_match = re.match(r'\[([^\]]+)\]\s+(\d{2}):(\d{2}):(\d{2})', line)

            if timestamp_match:
                speaker = timestamp_match.group(1)
                hours = int(timestamp_match.group(2))
                minutes = int(timestamp_match.group(3))
                seconds = int(timestamp_match.group(4))
                start_time = hours * 3600 + minutes * 60 + seconds

                # 次の行がテキスト内容
                i += 1
                if i < len(lines):
                    text = lines[i].strip()

                    if text:
                        # 次のタイムスタンプを探して終了時間を設定
                        end_time = start_time + 30  # デフォルト30秒
                        for j in range(i + 1, len(lines)):
                            next_line = lines[j].strip()
                            next_match = re.match(r'\[([^\]]+)\]\s+(\d{2}):(\d{2}):(\d{2})', next_line)
                            if next_match:
                                next_hours = int(next_match.group(2))
                                next_minutes = int(next_match.group(3))
                                next_seconds = int(next_match.group(4))
                                end_time = next_hours * 3600 + next_minutes * 60 + next_seconds
                                break

                        segments.append({
                            "start": start_time,
                            "end": end_time,
                            "speaker": speaker,
                            "text": text
                        })

            i += 1

        print(f"✅ 読み込み完了: {len(segments)}セグメント")
        return segments

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return []

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. 感情分析（Spark UDFによる並列処理）
# MAGIC
# MAGIC ### 使用モデル
# MAGIC - **Claude Sonnet 4.5** (`databricks-claude-sonnet-4-5`)
# MAGIC
# MAGIC ### 処理方式
# MAGIC - Spark UDFで各セグメントを並列処理
# MAGIC - クラスターのリソースを活用して高速分析
# MAGIC
# MAGIC ### 分析内容
# MAGIC 各発話について以下を分析：
# MAGIC - 感情ラベル（ポジティブ/ネガティブ/中立）
# MAGIC - 感情スコア（-1.0〜1.0）
# MAGIC - 信頼度（0.0〜1.0）

# COMMAND ----------

def analyze_emotion(text: str, timestamp: float) -> Dict:
    """感情分析"""
    prompt = f"""発言（{timestamp//60:.0f}分{timestamp%60:.0f}秒）の感情を分析:

{text}

JSON形式で回答してください:
{{
    "emotion": "ポジティブ/ネガティブ/中立",
    "sentiment_score": -1.0〜1.0,
    "confidence": 0.0〜1.0
}}"""

    try:
        response = fm_client.predict(
            endpoint="databricks-claude-sonnet-4-5",
            inputs={
                "messages": [
                    {"role": "system", "content": "あなたは感情分析の専門家です。テキストの感情を正確に分析してJSON形式で返してください。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            }
        )

        result_text = response['choices'][0]['message']['content']
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

        if json_match:
            return json.loads(json_match.group())
        else:
            return {"emotion": "中立", "sentiment_score": 0.0, "confidence": 0.5}
    except Exception as e:
        print(f"   分析エラー: {e}")
        return {"emotion": "中立", "sentiment_score": 0.0, "confidence": 0.0}

def analyze_all_segments(segments: List[Dict]) -> pd.DataFrame:
    """全セグメント分析（Spark UDFで並列化）"""
    print(f"🧠 感情分析: {len(segments)}セグメント（並列処理）")

    # PandasデータフレームをSparkデータフレームに変換
    segments_df = spark.createDataFrame(pd.DataFrame(segments))

    # UDF定義（感情分析）
    from pyspark.sql.functions import udf, struct
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType

    # 戻り値のスキーマ定義
    emotion_schema = StructType([
        StructField("emotion", StringType(), True),
        StructField("sentiment_score", DoubleType(), True)
    ])

    def analyze_emotion_udf(text: str, timestamp: float) -> dict:
        """UDF用の感情分析関数"""
        try:
            emotion = analyze_emotion(text, timestamp)
            return {
                "emotion": emotion.get('emotion', '中立'),
                "sentiment_score": float(emotion.get('sentiment_score', 0.0))
            }
        except Exception as e:
            print(f"分析エラー: {e}")
            return {"emotion": "中立", "sentiment_score": 0.0}

    # UDF登録
    emotion_udf = udf(analyze_emotion_udf, emotion_schema)

    # UDFを適用
    result_df = segments_df.withColumn(
        "emotion_result",
        emotion_udf(segments_df.text, segments_df.start)
    )

    # 結果を展開
    result_df = result_df.select(
        "start",
        "end",
        "speaker",
        "text",
        result_df.emotion_result.emotion.alias("emotion"),
        result_df.emotion_result.sentiment_score.alias("sentiment_score")
    )

    # Pandasデータフレームに変換して返す
    result_pdf = result_df.toPandas()
    result_pdf = result_pdf.rename(columns={"start": "start_time", "end": "end_time"})

    print("✅ 完了")
    return result_pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. 分析実行と基本可視化
# MAGIC
# MAGIC ### Step 1: 文字起こし読み込み
# MAGIC ファイルからセグメントデータを抽出
# MAGIC
# MAGIC ### Step 2: 感情分析
# MAGIC 全セグメントを並列処理で分析
# MAGIC
# MAGIC ### Step 3〜5: 可視化
# MAGIC - **時系列推移グラフ** - 発話者別に色分けした感情スコアの推移
# MAGIC - **感情分布（円グラフ）** - 全体および発話者別の感情割合
# MAGIC - **ヒストグラム** - 感情スコアの分布
# MAGIC - **箱ひげ図** - 発話者別の統計量比較
# MAGIC - **平均スコア棒グラフ** - 発話者別の平均感情スコア

# COMMAND ----------

# Step 1: 文字起こしファイルを読み込み
segments = load_transcript_from_file(transcript_path)
display(pd.DataFrame(segments).head(20))

# COMMAND ----------

# Step 2: 感情分析
emotion_df = analyze_all_segments(segments)
display(emotion_df)

# COMMAND ----------

# Step 3: 結果の可視化

# 感情の推移をプロット（発話者別）
fig = go.Figure()

# 発話者ごとに色分けしてプロット
speakers = emotion_df['speaker'].unique()
colors = px.colors.qualitative.Plotly

for i, speaker in enumerate(speakers):
    speaker_data = emotion_df[emotion_df['speaker'] == speaker]

    fig.add_trace(go.Scatter(
        x=speaker_data['start_time'],
        y=speaker_data['sentiment_score'],
        mode='lines+markers',
        name=speaker,
        line=dict(color=colors[i % len(colors)], width=2),
        marker=dict(size=8),
        hovertemplate='<b>発話者</b>: ' + speaker + '<br><b>時間</b>: %{x}秒<br><b>スコア</b>: %{text}<br><b>テキスト</b>: %{customdata}<extra></extra>',
        text=[f"{score:.2f}" for score in speaker_data['sentiment_score']],
        customdata=speaker_data['text']
    ))

# ゼロラインを追加
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

fig.update_layout(
    title='感情分析の時系列推移（発話者別）',
    xaxis_title='時間（秒）',
    yaxis_title='感情スコア',
    hovermode='closest',
    template='plotly_white',
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()

# COMMAND ----------

# Step 4: 感情分布の可視化（円グラフ - 全体）
emotion_counts = emotion_df['emotion'].value_counts()

fig2 = go.Figure(data=[go.Pie(
    labels=emotion_counts.index,
    values=emotion_counts.values,
    hole=0.3,
    marker=dict(colors=['#00CC96', '#EF553B', '#636EFA'])
)])

fig2.update_layout(
    title='感情分布（全体）',
    template='plotly_white',
    height=400
)

fig2.show()

# COMMAND ----------

# Step 4-2: 感情分布の可視化（発話者別）
from plotly.subplots import make_subplots

speakers = emotion_df['speaker'].unique()
n_speakers = len(speakers)

# サブプロットの行列数を計算
cols = min(3, n_speakers)
rows = (n_speakers + cols - 1) // cols

fig2_speaker = make_subplots(
    rows=rows,
    cols=cols,
    subplot_titles=[f"{speaker}" for speaker in speakers],
    specs=[[{"type": "pie"}] * cols for _ in range(rows)]
)

for i, speaker in enumerate(speakers):
    speaker_data = emotion_df[emotion_df['speaker'] == speaker]
    emotion_counts_speaker = speaker_data['emotion'].value_counts()

    row = i // cols + 1
    col = i % cols + 1

    fig2_speaker.add_trace(
        go.Pie(
            labels=emotion_counts_speaker.index,
            values=emotion_counts_speaker.values,
            hole=0.3,
            marker=dict(colors=['#00CC96', '#EF553B', '#636EFA'])
        ),
        row=row,
        col=col
    )

fig2_speaker.update_layout(
    title_text='感情分布（発話者別）',
    template='plotly_white',
    height=400 * rows,
    showlegend=True
)

fig2_speaker.show()

# COMMAND ----------

# Step 5: 感情スコアのヒストグラム（全体）
fig3 = px.histogram(
    emotion_df,
    x='sentiment_score',
    nbins=30,
    title='感情スコアの分布（全体）',
    labels={'sentiment_score': '感情スコア', 'count': '頻度'},
    template='plotly_white',
    color_discrete_sequence=['royalblue']
)

fig3.update_layout(height=400)
fig3.show()

# COMMAND ----------

# Step 5-2: 感情スコアのヒストグラム（発話者別）
fig4 = px.histogram(
    emotion_df,
    x='sentiment_score',
    color='speaker',
    nbins=30,
    title='感情スコアの分布（発話者別）',
    labels={'sentiment_score': '感情スコア', 'count': '頻度', 'speaker': '発話者'},
    template='plotly_white',
    barmode='overlay',
    opacity=0.7
)

fig4.update_layout(height=400)
fig4.show()

# COMMAND ----------

# Step 5-3: 発話者別の箱ひげ図
fig5 = px.box(
    emotion_df,
    x='speaker',
    y='sentiment_score',
    title='発話者別の感情スコア分布（箱ひげ図）',
    labels={'speaker': '発話者', 'sentiment_score': '感情スコア'},
    template='plotly_white',
    color='speaker'
)

fig5.update_layout(height=400)
fig5.show()

# COMMAND ----------

# Step 5-4: 発話者別の平均感情スコア（棒グラフ）
speaker_avg = emotion_df.groupby('speaker')['sentiment_score'].mean().reset_index()
speaker_avg = speaker_avg.sort_values('sentiment_score', ascending=False)

fig6 = px.bar(
    speaker_avg,
    x='speaker',
    y='sentiment_score',
    title='発話者別の平均感情スコア',
    labels={'speaker': '発話者', 'sentiment_score': '平均感情スコア'},
    template='plotly_white',
    color='sentiment_score',
    color_continuous_scale=['red', 'yellow', 'green']
)

fig6.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
fig6.update_layout(height=400)
fig6.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. 感情変動分析 - 原因となった発話の特定
# MAGIC
# MAGIC ### 分析目的
# MAGIC 発話者の感情が急激に変化した際、その直前にあった他者の発言を特定することで、
# MAGIC **誰の発言が誰の感情にどのような影響を与えたか**を明らかにします。
# MAGIC
# MAGIC ### 分析方法
# MAGIC 1. 各発話者について連続する発話間の感情スコア変化を計算
# MAGIC 2. 閾値（デフォルト0.3）を超える変動を検出
# MAGIC 3. その間に発生した他の発話者の発言を「トリガー発話」として特定
# MAGIC 4. 改善/悪化の種類と変化量を記録
# MAGIC
# MAGIC ### 可視化内容
# MAGIC - **変動量の棒グラフ** - 改善/悪化別に発話者ごとの変動を表示
# MAGIC - **影響力分析** - どの発話者が他者の感情に最も影響を与えているか
# MAGIC - **詳細イベント表示** - 変化前・トリガー・変化後の発言を時系列で表示

# COMMAND ----------

# Step 6: 感情変動分析 - 原因となった発話の特定

def analyze_emotion_changes(emotion_df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    感情の急激な変動を検出し、その原因となった可能性のある発話を特定

    Args:
        emotion_df: 感情分析結果のデータフレーム
        threshold: 感情変動とみなす閾値（デフォルト: 0.3）

    Returns:
        感情変動の分析結果
    """
    changes = []

    # 発話者ごとに分析
    for speaker in emotion_df['speaker'].unique():
        speaker_data = emotion_df[emotion_df['speaker'] == speaker].sort_values('start_time').reset_index(drop=True)

        for i in range(1, len(speaker_data)):
            current = speaker_data.iloc[i]
            previous = speaker_data.iloc[i-1]

            # 感情スコアの変化量を計算
            score_change = current['sentiment_score'] - previous['sentiment_score']

            # 閾値を超える変動があった場合
            if abs(score_change) >= threshold:
                # その間に他の発話者の発言を探す
                between_time_start = previous['start_time']
                between_time_end = current['start_time']

                # 他の発話者の発言を取得
                other_speakers = emotion_df[
                    (emotion_df['speaker'] != speaker) &
                    (emotion_df['start_time'] > between_time_start) &
                    (emotion_df['start_time'] < between_time_end)
                ].sort_values('start_time')

                if len(other_speakers) > 0:
                    # 最も近い発言を特定
                    trigger_utterance = other_speakers.iloc[-1]  # 直前の発言

                    change_type = "改善" if score_change > 0 else "悪化"

                    changes.append({
                        "affected_speaker": speaker,
                        "change_type": change_type,
                        "score_change": score_change,
                        "before_score": previous['sentiment_score'],
                        "after_score": current['sentiment_score'],
                        "before_time": previous['start_time'],
                        "after_time": current['start_time'],
                        "before_text": previous['text'],
                        "after_text": current['text'],
                        "trigger_speaker": trigger_utterance['speaker'],
                        "trigger_time": trigger_utterance['start_time'],
                        "trigger_text": trigger_utterance['text'],
                        "trigger_emotion": trigger_utterance['emotion'],
                        "trigger_score": trigger_utterance['sentiment_score']
                    })

    return pd.DataFrame(changes)

# 感情変動を分析
emotion_changes_df = analyze_emotion_changes(emotion_df, threshold=0.3)

if len(emotion_changes_df) > 0:
    print(f"✅ {len(emotion_changes_df)}件の感情変動を検出")
    display(emotion_changes_df)
else:
    print("感情の急激な変動は検出されませんでした")

# COMMAND ----------

# Step 6-2: 感情変動の可視化

if len(emotion_changes_df) > 0:
    # 改善と悪化に分けて集計
    improvements = emotion_changes_df[emotion_changes_df['change_type'] == '改善']
    deteriorations = emotion_changes_df[emotion_changes_df['change_type'] == '悪化']

    # 変動の種類別集計
    fig7 = go.Figure()

    if len(improvements) > 0:
        fig7.add_trace(go.Bar(
            name='感情改善',
            x=improvements['affected_speaker'],
            y=improvements['score_change'],
            marker_color='green',
            hovertemplate='<b>影響を受けた人</b>: %{x}<br><b>変化量</b>: %{y:.2f}<br><extra></extra>'
        ))

    if len(deteriorations) > 0:
        fig7.add_trace(go.Bar(
            name='感情悪化',
            x=deteriorations['affected_speaker'],
            y=deteriorations['score_change'],
            marker_color='red',
            hovertemplate='<b>影響を受けた人</b>: %{x}<br><b>変化量</b>: %{y:.2f}<br><extra></extra>'
        ))

    fig7.update_layout(
        title='発話者別の感情変動',
        xaxis_title='影響を受けた発話者',
        yaxis_title='感情スコア変化量',
        template='plotly_white',
        height=400,
        barmode='group'
    )

    fig7.show()

    # COMMAND ----------

    # Step 6-3: トリガー発話者の影響力分析
    trigger_impact = emotion_changes_df.groupby('trigger_speaker').agg({
        'score_change': ['count', 'mean', 'sum']
    }).reset_index()

    trigger_impact.columns = ['trigger_speaker', 'count', 'avg_impact', 'total_impact']
    trigger_impact = trigger_impact.sort_values('total_impact', ascending=False)

    fig8 = go.Figure()

    fig8.add_trace(go.Bar(
        x=trigger_impact['trigger_speaker'],
        y=trigger_impact['total_impact'],
        marker_color=trigger_impact['total_impact'],
        marker_colorscale='RdYlGn',
        marker_cmid=0,
        text=trigger_impact['count'],
        texttemplate='%{text}回',
        textposition='outside',
        hovertemplate='<b>発話者</b>: %{x}<br><b>影響力合計</b>: %{y:.2f}<br><b>影響回数</b>: %{text}<extra></extra>'
    ))

    fig8.update_layout(
        title='発話者別の影響力（他者の感情変動への寄与度）',
        xaxis_title='トリガー発話者',
        yaxis_title='感情変動の合計影響',
        template='plotly_white',
        height=400
    )

    fig8.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig8.show()

    # COMMAND ----------

    # Step 6-4: 重要な感情変動イベントの詳細表示
    print("📊 重要な感情変動イベント")
    print("="*80)

    # 改善イベント（スコア上昇が大きい順）
    improvements_top = emotion_changes_df[emotion_changes_df['change_type'] == '改善'].nlargest(5, 'score_change', keep='all')

    if len(improvements_top) > 0:
        print("\n✅ 【感情改善イベント】（変化量上位5件）")
        print("-"*80)

        for idx, row in improvements_top.iterrows():
            print(f"\n影響を受けた人: {row['affected_speaker']}")
            print(f"  変化: {row['before_score']:.2f} → {row['after_score']:.2f} (差分: {row['score_change']:+.2f})")
            print(f"  時間: {int(row['before_time']//60)}:{int(row['before_time']%60):02d} → {int(row['after_time']//60)}:{int(row['after_time']%60):02d}")
            print(f"\n  🔹 変化前の発言:")
            print(f"    {row['before_text'][:100]}...")
            print(f"\n  ⚡ トリガーとなった発言 ({row['trigger_speaker']}):")
            print(f"    [{int(row['trigger_time']//60)}:{int(row['trigger_time']%60):02d}] {row['trigger_text'][:100]}...")
            print(f"    感情: {row['trigger_emotion']} (スコア: {row['trigger_score']:.2f})")
            print(f"\n  🔹 変化後の発言:")
            print(f"    {row['after_text'][:100]}...")
            print("-"*80)

    # 悪化イベント（スコア下降が大きい順）
    deteriorations_top = emotion_changes_df[emotion_changes_df['change_type'] == '悪化'].nsmallest(5, 'score_change', keep='all')

    if len(deteriorations_top) > 0:
        print("\n\n❌ 【感情悪化イベント】（変化量上位5件）")
        print("-"*80)

        for idx, row in deteriorations_top.iterrows():
            print(f"\n影響を受けた人: {row['affected_speaker']}")
            print(f"  変化: {row['before_score']:.2f} → {row['after_score']:.2f} (差分: {row['score_change']:+.2f})")
            print(f"  時間: {int(row['before_time']//60)}:{int(row['before_time']%60):02d} → {int(row['after_time']//60)}:{int(row['after_time']%60):02d}")
            print(f"\n  🔹 変化前の発言:")
            print(f"    {row['before_text'][:100]}...")
            print(f"\n  ⚡ トリガーとなった発言 ({row['trigger_speaker']}):")
            print(f"    [{int(row['trigger_time']//60)}:{int(row['trigger_time']%60):02d}] {row['trigger_text'][:100]}...")
            print(f"    感情: {row['trigger_emotion']} (スコア: {row['trigger_score']:.2f})")
            print(f"\n  🔹 変化後の発言:")
            print(f"    {row['after_text'][:100]}...")
            print("-"*80)

# COMMAND ----------

# Step 7: 発話者別の感情分析サマリー
print("📊 感情分析サマリー")
print("="*60)
print(f"総セグメント数: {len(emotion_df)}")
print(f"発話者数: {emotion_df['speaker'].nunique()}")

print(f"\n感情分布:")
print(emotion_df['emotion'].value_counts())

print(f"\n平均感情スコア: {emotion_df['sentiment_score'].mean():.3f}")
print(f"最高スコア: {emotion_df['sentiment_score'].max():.3f}")
print(f"最低スコア: {emotion_df['sentiment_score'].min():.3f}")

# ポジティブ/ネガティブの割合
positive_pct = (emotion_df['sentiment_score'] > 0).sum() / len(emotion_df) * 100
negative_pct = (emotion_df['sentiment_score'] < 0).sum() / len(emotion_df) * 100
neutral_pct = (emotion_df['sentiment_score'] == 0).sum() / len(emotion_df) * 100

print(f"\nポジティブ: {positive_pct:.1f}%")
print(f"ネガティブ: {negative_pct:.1f}%")
print(f"中立: {neutral_pct:.1f}%")

# 発話者別サマリー
print("\n" + "="*60)
print("📊 発話者別サマリー")
print("="*60)

for speaker in emotion_df['speaker'].unique():
    speaker_data = emotion_df[emotion_df['speaker'] == speaker]
    print(f"\n【{speaker}】")
    print(f"  発話回数: {len(speaker_data)}")
    print(f"  平均感情スコア: {speaker_data['sentiment_score'].mean():.3f}")
    print(f"  感情分布: {dict(speaker_data['emotion'].value_counts())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 分析完了！
# MAGIC
# MAGIC ### 実装機能まとめ
# MAGIC
# MAGIC #### 📄 データ処理
# MAGIC - ✅ タイムスタンプ付き文字起こしの自動パース
# MAGIC - ✅ 発話者情報と時刻の抽出
# MAGIC - ✅ セグメント単位でのデータ構造化
# MAGIC
# MAGIC #### 🤖 感情分析
# MAGIC - ✅ Claude Sonnet 4.5による高精度分析
# MAGIC - ✅ Spark UDFによる並列処理（高速化）
# MAGIC - ✅ 感情ラベル・スコア・信頼度の抽出
# MAGIC
# MAGIC #### 📊 基本可視化
# MAGIC - ✅ 時系列推移グラフ（発話者別色分け）
# MAGIC - ✅ 感情分布の円グラフ（全体＋発話者別）
# MAGIC - ✅ ヒストグラム（全体＋発話者別重ね表示）
# MAGIC - ✅ 箱ひげ図（発話者別統計量比較）
# MAGIC - ✅ 平均スコア棒グラフ
# MAGIC
# MAGIC #### 🔍 高度な分析
# MAGIC - ✅ **感情変動検出** - 急激な感情変化の特定
# MAGIC - ✅ **トリガー発話の特定** - 変動の原因となった発言を特定
# MAGIC - ✅ **影響力分析** - 発話者別の感情的影響力を定量化
# MAGIC - ✅ **詳細イベント表示** - 重要な変動の文脈を可視化
# MAGIC
# MAGIC #### 📈 統計サマリー
# MAGIC - ✅ 全体統計（総セグメント数、感情分布、平均スコア）
# MAGIC - ✅ 発話者別統計（発話回数、平均スコア、感情分布）
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### 活用例
# MAGIC - **会議分析**: 参加者の感情推移とその要因を特定
# MAGIC - **カスタマーサポート**: 顧客満足度の変化点を分析
# MAGIC - **インタビュー分析**: 話者間の相互作用を可視化
# MAGIC - **チームダイナミクス**: メンバー間の影響関係を理解