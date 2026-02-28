# Comment Obtenir les News sur Apple

Ce guide explique les différentes méthodes pour récupérer des news spécifiques à Apple (ou toute autre entreprise).

## 🎯 Méthodes Disponibles

### 1. **yfinance** (GRATUIT - RECOMMANDÉ ✅)

La méthode la plus simple et gratuite pour obtenir les dernières news sur Apple.

```python
from src.ingestion import DataIngestion

ingestion = DataIngestion()
news = ingestion.fetch_news_from_yfinance('AAPL')

# Afficher les news
for idx, row in news.iterrows():
    print(f"{row['title']} - {row['publisher']}")
```

**Avantages:**
- ✅ Gratuit
- ✅ Pas besoin de clé API
- ✅ Facile à utiliser
- ✅ News récentes et pertinentes

**Limitations:**
- ⚠️ Limité aux dernières news (généralement ~10-20 articles)
- ⚠️ Pas d'historique profond

---

### 2. **Filtrage du Dataset Kaggle**

Si vous avez téléchargé le dataset Kaggle, vous pouvez filtrer les articles contenant des mots-clés liés à Apple.

```python
from src.ingestion import DataIngestion

ingestion = DataIngestion()

# Définir les mots-clés Apple
keywords = ['Apple', 'AAPL', 'iPhone', 'iPad', 'Mac', 'Tim Cook']

# Charger et filtrer
news_df = ingestion.load_kaggle_financial_news()
apple_news = ingestion.filter_news_by_company(news_df, keywords)
```

**Avantages:**
- ✅ Grand volume de données historiques
- ✅ Gratuit après téléchargement
- ✅ Données labellisées (sentiment)

**Limitations:**
- ⚠️ Nécessite téléchargement manuel depuis Kaggle
- ⚠️ Données pas toujours à jour
- ⚠️ Peut ne pas contenir beaucoup d'articles sur Apple spécifiquement

---

### 3. **NewsAPI** (Clé API Gratuite)

Pour des recherches plus avancées avec historique.

```python
from src.ingestion import DataIngestion

ingestion = DataIngestion()

# Obtenir une clé gratuite sur https://newsapi.org/
API_KEY = "votre_cle_api"

news = ingestion.fetch_news_from_newsapi(
    query='Apple OR AAPL OR iPhone',
    api_key=API_KEY,
    from_date='2024-01-01',
    to_date='2024-12-31'
)
```

**Avantages:**
- ✅ Recherche avancée avec dates
- ✅ Grand nombre de sources
- ✅ 100 requêtes/jour gratuites
- ✅ Contenu complet des articles

**Limitations:**
- ⚠️ Nécessite inscription (gratuite)
- ⚠️ Limite de 100 articles par requête
- ⚠️ Historique limité à 1 mois (version gratuite)

**Comment obtenir une clé:**
1. Aller sur https://newsapi.org/
2. Créer un compte (gratuit)
3. Copier votre clé API
4. Utiliser la clé dans votre code

---

### 4. **Méthode Complète (Tout-en-un)** ⭐

La méthode recommandée qui combine toutes les sources:

```python
from src.ingestion import DataIngestion

ingestion = DataIngestion()

# Récupère TOUT: données de marché + news
apple_data = ingestion.fetch_company_data(
    ticker='AAPL',
    company_keywords=['Apple', 'AAPL', 'iPhone'],
    market_period='1y',
    use_newsapi=False,  # Mettre True si vous avez une clé
    newsapi_key=None    # Votre clé si use_newsapi=True
)

# Accéder aux différentes données
market_data = apple_data['market_data']          # Cours de l'action
news_yf = apple_data['news_yfinance']           # News de yfinance
news_kaggle = apple_data['news_filtered']       # News filtrées Kaggle
news_api = apple_data['news_api']               # News de NewsAPI
```

---

## 🚀 Quick Start - Exemple Apple

Lancez simplement:

```bash
python examples/apple_analysis_example.py
```

Ce script va:
1. ✅ Récupérer 1 an de données de marché Apple
2. ✅ Récupérer les dernières news Apple via yfinance
3. ✅ Filtrer le dataset Kaggle pour les news Apple (si disponible)
4. ✅ Sauvegarder toutes les données dans `data/processed/`

---

## 📊 Données Obtenues

### Données de Marché (market_data)
```
Date       | Open    | High    | Low     | Close   | Volume
2024-01-01 | 185.00  | 187.50  | 184.20  | 186.75  | 50M
...
```

### News (news_yfinance)
```
title                              | publisher | published           | link
-----------------------------------|-----------|---------------------|-----
Apple Announces New iPhone 15      | CNBC      | 2024-02-15 10:30:00 | https://...
Tim Cook Discusses AI Strategy     | Bloomberg | 2024-02-14 14:20:00 | https://...
...
```

---

## 💡 Recommandations

### Pour débuter (GRATUIT):
```python
# Méthode 1: yfinance seulement
news = ingestion.fetch_news_from_yfinance('AAPL')
market = ingestion.fetch_market_data(['AAPL'], period='1y')
```

### Pour plus de données:
```python
# Méthode 1 + 2: yfinance + Kaggle
apple_data = ingestion.fetch_company_data(
    ticker='AAPL',
    company_keywords=['Apple', 'AAPL', 'iPhone'],
    market_period='1y'
)
```

### Pour analyse approfondie:
```python
# Toutes les sources (yfinance + Kaggle + NewsAPI)
apple_data = ingestion.fetch_company_data(
    ticker='AAPL',
    company_keywords=['Apple', 'AAPL', 'iPhone', 'iPad'],
    market_period='2y',
    use_newsapi=True,
    newsapi_key='VOTRE_CLE'
)
```

---

## 🔧 Installation

Installez les dépendances:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

- **yfinance** est la source la plus fiable et gratuite pour les news récentes
- **Kaggle** est utile pour l'historique et les sentiments labellisés
- **NewsAPI** offre plus de flexibilité mais nécessite une clé
- Toutes les données sont automatiquement sauvegardées dans `data/processed/`

---

## 🆘 Problèmes Fréquents

### "No news found for AAPL"
→ Normal, yfinance peut ne pas toujours avoir de news. Essayez plus tard ou utilisez NewsAPI.

### "No Kaggle dataset found"
→ Téléchargez le dataset: https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news
→ Placez-le dans `data/raw/`

### "NewsAPI rate limit exceeded"
→ Version gratuite limitée à 100 requêtes/jour
→ Attendez 24h ou passez à la version payante
