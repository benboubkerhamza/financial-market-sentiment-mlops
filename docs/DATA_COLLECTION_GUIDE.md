# Quick Start Guide - Data Collection

## Prerequisites

Before running the data collection, you need:

### 1. Kaggle Financial News Dataset

**Option A: Manual Download (Recommended)**
1. Visit: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
2. Click "Download" (requires free Kaggle account)
3. Unzip the downloaded file
4. Move `all-data.csv` to: `data/raw/all-data.csv`

**Option B: Using Kaggle API**
```powershell
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d ankurzing/sentiment-analysis-for-financial-news -p data/raw --unzip
```

### 2. NewsAPI Token (Optional)

NewsAPI provides recent financial news articles.

1. Visit: https://newsapi.org/register
2. Sign up (free, no credit card required)
3. Copy your API key
4. Create `.env` file:
   ```bash
   cp .env.example .env
   ```
5. Add your key to `.env`:
   ```
   NEWSAPI_KEY=your_actual_key_here
   ```

**Free Plan Limits:**
- 100 requests/day
- Last 30 days of articles only
- Perfect for development

---

## Data Collection Options

### Option 1: Basic Collection (No NewsAPI)

Collect Kaggle news + market data only:

```powershell
# Run collection script
python scripts/collect_data.py
```

This will fetch:
- Kaggle Financial News dataset (4,846 articles)
- Market data for 10 major stocks (2 years)

### Option 2: Full Collection (With NewsAPI)

Modify `scripts/collect_data.py` to include NewsAPI, or use targeted collection:

```python
from src.ingestion import DataIngestion

ingestion = DataIngestion()

# Get all data for Apple including recent news
data = ingestion.fetch_company_data(
    ticker='AAPL',
    company_keywords=['Apple', 'AAPL', 'iPhone', 'Tim Cook'],
    market_period='2y',
    use_newsapi=True,
    newsapi_key='your_key_here'
)
```

### Option 3: Use Existing Examples

```powershell
# Apple analysis (no NewsAPI required)
python examples/apple_analysis_example.py

# General ingestion
python examples/data_ingestion_example.py
```

---

## Verify Data Collection

After running collection, check:

```powershell
# List collected files
ls data/processed/

# Expected files:
# - news_kaggle.csv (Kaggle dataset)
# - market_data_AAPL.csv (Apple market data)
# - market_data_MSFT.csv (Microsoft market data)
# - ... (other tickers)
```

---

## Troubleshooting

### "Kaggle dataset not found"
- Ensure `all-data.csv` is in `data/raw/`
- Check file name is exactly `all-data.csv`

### "NewsAPI rate limit exceeded"
- Free plan: 100 requests/day
- Wait 24 hours or upgrade plan
- Continue without NewsAPI (Kaggle + yfinance still work)

### "yfinance connection error"
- Check internet connection
- Some tickers might be delisted
- Retry after a few minutes

---

## Next Steps

Once data is collected:

1. **Verify data quality**:
   ```powershell
   python -c "import pandas as pd; df=pd.read_csv('data/processed/news_kaggle.csv'); print(df.info())"
   ```

2. **Start preprocessing**: See Phase 2 in [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md)

3. **Exploratory analysis**: Create notebooks in `notebooks/` folder

4. **Commit your work**:
   ```powershell
   git add scripts/ .env.example
   git commit -m "docs: Add data collection script and setup guide"
   git push origin dev
   ```
