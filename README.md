# CodeAlpha Internship — Task 4: Sentiment Analysis

## What This Does
Analyzes Amazon-style product reviews and classifies them as **Positive**, **Negative**, or **Neutral** using NLP (TextBlob library). Also generates word clouds and visual dashboards.

## Libraries Required
```bash
pip install textblob wordcloud pandas numpy matplotlib seaborn
```

## Output Files
- `task4_sentiment_analysis.png` — Main dashboard (4 charts)
- `task4_wordclouds.png` — Word clouds for each sentiment

## What the Code Covers (as per task)
- ✅ Classify text as Positive, Negative, or Neutral
- ✅ NLP techniques using TextBlob (polarity scoring)
- ✅ Analysis on Amazon-style product reviews
- ✅ Understand public opinion through sentiment patterns
- ✅ Insights for marketing and product development

## Charts in Dashboard
| Chart | What It Shows |
|---|---|
| Pie Chart | Overall sentiment distribution |
| Bar Chart | Count of each sentiment |
| Histogram | Polarity score distribution |
| Stacked Bar | Sentiment breakdown by product |
| Word Clouds | Most common words per sentiment |

## Key Results
- Polarity score ranges from -1 (very negative) to +1 (very positive)
- Word clouds reveal what customers talk about most
- Product-wise analysis helps identify which products need improvement

Upload these files:
- `task4_sentiment_analysis.py`
- `task4_sentiment_analysis.png`
- `task4_wordclouds.png`
- This README
