# sentiment-analysis

Sentiment Analysis on Product and Business Reviews

Using VADER, RoBERTa, and BERT Models

1. Project Overview

This project presents an end-to-end Natural Language Processing (NLP) pipeline for sentiment analysis and rating prediction using both traditional lexicon-based methods and modern Transformer-based deep learning models.

The goal is to evaluate how effectively textual reviews can be used to:
- Infer overall sentiment
- Predict numerical star ratings
- Analyze public opinion toward products and businesses

The system is tested on:
- Amazon product reviews
- Real-world Yelp restaurant reviews
- User-provided reviews in real time

2. Objectives
- Compare classical sentiment analysis techniques with Transformer-based models
- Evaluate sentiment polarity alignment with user-provided star ratings
- Predict numerical ratings directly from text
- Demonstrate real-world applicability using scraped online reviews
- Provide a reusable framework for sentiment analysis tasks

3. Models and Techniques
3.1 VADER (NLTK)
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based sentiment analyzer included in the NLTK library.

Characteristics:
- Lexicon and rule-based approach
- Outputs positive, neutral, negative, and compound scores
- Efficient and interpretable
- Does not consider contextual relationships between words

3.2 RoBERTa (Transformer-Based Sentiment Analysis)
A pretrained RoBERTa model fine-tuned on Twitter data is used for contextual sentiment analysis.

Model:
cardiffnlp/twitter-roberta-base-sentiment

Advantages:
- Captures context and semantic relationships
- Handles sarcasm and mixed sentiment more effectively
- Produces probability scores for positive, neutral, and negative sentiment

3.3 BERT (Star Rating Prediction)
A multilingual BERT-based model is used to predict numerical star ratings directly from text.

Model:
nlptown/bert-base-multilingual-uncased-sentiment

Output:
- Predicted rating between 1 and 5 stars

This allows quantitative evaluation of how closely sentiment aligns with user ratings.

4. Dataset Description
4.1 Amazon Reviews Dataset
Source: Amazon Fine Food Reviews
Fields used:
- Score (1–5 star rating)
- Summary
- Text

Dataset is downsampled to 500 reviews for computational efficiency
Summary and text fields are combined into a single review field

4.2 Yelp Reviews
- Reviews scraped directly from Yelp business pages
- Used to test the generalization of models on unseen, real-world data
- Demonstrates business-level sentiment estimation

5. Project Structure
.
├── dataset/
│   └── Reviews.csv                 # Original dataset (optional)
├── amazon_reviews_dataset_500.csv  # Sampled dataset
├── sentiment_analysis.py           # Main analysis script
├── README.md                       # Project documentation
└── requirements.txt                # Project dependencies

6. Methodology
1. Data loading and preprocessing
2. Exploratory data analysis and visualization
3. Tokenization and part-of-speech tagging
4. Sentiment analysis using VADER
5. Contextual sentiment analysis using RoBERTa
6. Comparative analysis of sentiment scores vs star ratings
7. Star rating prediction using BERT
8. Model evaluation using accuracy metrics
9. Yelp review scraping and sentiment inference
10. Real-time sentiment and rating prediction

7. Evaluation Metrics
7.1 Star Rating Prediction Accuracy
- Exact rating prediction accuracy: approximately 70 percent
- Prediction within ±1 star: approximately 95 percent
These results demonstrate strong alignment between textual sentiment and numerical ratings.

8. Key Observations
- VADER performs well on clear, sentiment-heavy text
- RoBERTa significantly improves performance on nuanced and mixed-sentiment reviews
- Transformer-based models show a strong correlation with actual user ratings
- Text alone can provide reliable insight into customer satisfaction

9. Real-Time Sentiment Analysis
The project includes an interactive command-line interface allowing users to enter custom reviews and receive:
-Predicted star rating (1–5)
- Sentiment probabilities (positive, neutral, negative)
This feature enables live experimentation and rapid sentiment assessment.

10. Installation and Setup
10.1 Clone the Repository
git clone https://github.com/your-username/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp

10.2 Install Dependencies
pip install -r requirements.txt

10.3 Download NLTK Resources (If Required)
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

11. Dependencies
- Python 3.8 or higher
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- transformers
- torch
- tqdm
- beautifulsoup4
- requests
- scikit-learn

12. Applications
- Product review analysis
- Customer feedback evaluation
- Business reputation monitoring
- Market research and sentiment tracking
- Recommendation systems
- Social media analytics

13. Limitations
- Pretrained models may not perfectly generalize to all domains
- Long reviews are truncated due to model input size constraints
- Sentiment does not always align perfectly with star ratings due to user subjectivity

14. Future Work
- Domain-specific fine-tuning of Transformer models
- Aspect-based sentiment analysis
- Deployment as a web application (Streamlit or FastAPI)
- Multilingual sentiment comparison
- Temporal sentiment trend analysis

15. Acknowledgements
- NLTK
- Hugging Face
- Amazon Fine Food Reviews Dataset
- Yelp
