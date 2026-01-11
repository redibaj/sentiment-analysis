# - VADER Model from NLTK
# - RoBERTa Pretrained Model from Hugging face 🤗
#storing and manipulating data
import pandas as pd
import numpy as np

#plotting
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm

#natural language tool kit
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#RoBERTa Model from Hugging Face
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

class CFG:
    load_complete_dataset = False
    install_dependencies = False
    running_on_kaggle = False


# ## Installs
if CFG.install_dependencies:
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('maxent_ne_chunker_tab')

# ## Load / Read Data
if CFG.running_on_kaggle:
    data_dir = '/kaggle/input/amazon-fine-food-reviews/Reviews.csv'
else:
    data_dir = 'dataset/Reviews.csv'
if CFG.load_complete_dataset:
    df = pd.read_csv(data_dir, usecols=['Score', 'Summary', 'Text'])
    print(df.shape) 
    print(df.head())

if CFG.running_on_kaggle:
    sampled_dataset_path = '/kaggle/input/amazon-review-dataset-500/amazon_reviews_dataset_500.csv'
else:
    sampled_dataset_path = 'amazon_reviews_dataset_500.csv'
    
if CFG.load_complete_dataset:
    #complete dataset has over half a million reviews 
    #the overall dataset is too large so we will downsample the data to 500 examples for analysis
    df = df.sample(500)
    
    # combine the summary / heading and text description
    df['review_text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')

    #we can also remove the previous original columns
    df.drop(['Summary', 'Text'], axis = 1, inplace = True)
    
    df = df.reset_index()
    df.drop(['index'], axis = 1, inplace=True)
    
    #and save the new data into csv
    df.to_csv(sampled_dataset_path, index = False)
else:
    df = pd.read_csv(sampled_dataset_path)

df.head()

fig, ax = plt.subplots(figsize = (10, 5))
ax = df['Score'].value_counts().sort_index().plot(kind = 'bar')
ax.set_xlabel('Review Stars')
plt.title('count of reviews by stars')
plt.show()

# The overall distribution of reviews seems to be more positive and then there is a slight spike in the 1 star negetive reviews, so data is mostly very polar.

# ## Basic NLTK

example = df['review_text'][50]
print(example)

#use the nltk's word tokenizer to smartly split the sentence into tokens 
#which can be used to understand the language by the computer
tokens = nltk.word_tokenize(example)
tokens[:8]

#find the part of speech for each of the token words with nltk
#each word will get a part of speech value associated with them 
#and they are represented with codes like NN is for singular nouns
tagged = nltk.pos_tag(tokens)
tagged[:10]

#we can group these tagged values into chunks of tags entities
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# ## VADER Sentiment Scoring 

# **VADER:** Valence Aware Dictonary and sEntiment Reasoner
# 
# We'll use NLTK's `SentimentInetensityAnalyzer` to get the negetive (neg), neutral (neu) and positive (pos) scores for the text.
# 
# **Approach:**
# - Uses a "bag of words" approach
# - Stop words are removed - words like (is, the, am, etc.)
# - each word is scored and combined to a total score to get a overall sentiment score of text.
# 
# **Limitations:** 
# - does not account for relationships between words, which is an important part of human speech.

sia = SentimentIntensityAnalyzer()

#example of using sentiment analyzer on a positive sentence
sia.polarity_scores('I am so happy! Today is such a great day and I love ML.')

#on the other hand sentiment analyzer on a negetive example
sia.polarity_scores('This is the worst thing ever, I would never buy something this bad.')

#It can also work well on some confusing sentences - to give a overall slightly positive rating for this sentence.
sia.polarity_scores('hey, I hate you, just kidding you know I love you!')


# **Observations:** \
# We are successfully able to identify the overall sentiment of the text with compound score which varies between 1 (for positive) and -1 (for negetive).
# 
# Now we can try to apply this to our reviews dataset.

#Run Polarity Score on the entire dataset
res = {}
for idx, row in df.iterrows():
    text = row['review_text']
    res[idx] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.merge(df, how = 'left', right_index=True, left_index=True)
vaders.head()


# **How to measure effectiveness?**
# 
# One way to see if our model is working as expected or not, is to check an assumption.
# 
# We can assume that what our model classifies as more positive has higher star review \
# and the reviews predicted to be more negetive have a lower star review associated with them.
# 
# If this is true then, we can rely on text sentiment analysis to tell us how our users feel about products overall.

# Plot Vaders results
sns.barplot(data=vaders, x='Score', y='compound')
plt.title('Compound Score with respect to Amazon star reviews')
plt.show()


# **Observation:** 
# We can see that the assumptions seems to be right as higher star reviews have a high compound score associated with them.

# **Taking a look at each component of sentiment analysis (pos, neu and neg) with respect to star reviews**

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.show()


# **Observations:**
# - postivity score is higher as star reviews are higher star reviews
# - neural scores do not vary much accross different star reviews
# - negativity score decreases with higher star reviews
# 
# **So we can conclude that our sentiment analysis score does relate to the star reviews given by the users**

# **Limitations**
# 
# This way of analyzing text with vader does not consider the relationship between the words and the context in which they are used, so we can improve upon that using the state of the art Transformer based BERT models which can understand and learn context and make better prediction of overall sentiment.

# ## RoBERTa Pretrained Model

# - Use a model trained on large corpus of data (with hundred Millions+ examples) 
# - Transformer model accounts for the words bit also their context related to other words.
#     - like for sentences which have negative words in it but it actually is just sarcastic.
#     - or ironic sentences with positive words but overall negative sentiment.
#     
# We are essentially using transfer learning by using pretrained weights of a model created for analyzing the sentiments of tweets to perform review sentiment analysis.

model_name = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# **Comparing models performence over example review**

#VADER Results on example
print(example)
sia.polarity_scores(example)

#Run Roberta model for example
#here tf = tensorflow
encoded_text = tokenizer(example, return_tensors='pt') 
encoded_text

output = model(**encoded_text)

scores = output[0][0].detach().numpy()
scores = softmax(scores) 
scores_dict = {
    'roberta_pos': scores[0],
    'roberta_neu': scores[1],
    'roberta_neg': scores[2],
}
scores_dict

print(f"User's actual given star review on the example: {df['Score'][50]}")


# **Observation:**
# 
# The Roberta Model does seem to better understand the context and give a overall more appropriate score for the given example, that the user does start with a positive comment but actually have some complaints about the products too. But as we can confirm from user's given rating the overall sentiment and review is still positive.

# **Run the RoBERTa model over all the review examples**

def polarity_scores_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
    }
    
    return scores_dict


# In[27]:


results = {}
for idx, row in tqdm(df.iterrows(), total=len(df)):

    text = row['review_text']

    #Roberta Model can only accept text with max length of 514
    if len(text) > 514:
        text = text[:514]

    vader_result = sia.polarity_scores(text)    
    #rename the results of vader
    vader_result_rename = {}
    for key, value in vader_result.items():
        vader_result_rename[f'vader_{key}'] = value

    #making predictions using roberta model
    roberta_result = polarity_scores_roberta(text)

    #combine both results 
    result = vader_result_rename | roberta_result

    results[idx] = result


# In[28]:


results_df = pd.DataFrame(results).T
results_df = results_df.merge(df, how = 'left', right_index=True, left_index=True)
results_df.head()


# In[29]:


# Plot Roberta results
fig, axs = plt.subplots(1, 3, figsize=(30, 10))
sns.barplot(data=results_df, x='Score', y='roberta_pos', ax=axs[0])
sns.barplot(data=results_df, x='Score', y='roberta_neu', ax=axs[1])
sns.barplot(data=results_df, x='Score', y='roberta_neg', ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.show()


# **Observations:**
# - with roberta model there seems to be a even more clear trend as when positivity score increases the star rating increases and with higher star reviews negetive sentiment decreases drastically. So we can get a much better idea of how users are feeling with text sentiment prediction by Roberta Model.

# ## Review Examples

# **Positive Sentiment Prediction but low star review**
# 
# Which are the examples that our models classifies as having a positive sentiment but it actually has a low star rating

# In[30]:


results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['review_text'].values[0]


# - So we can see this is a confusing statement as the user praises Amazon's service but is actually disappointed by the product, and our model was not able to make that distinction.

# In[31]:


results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['review_text'].values[0]


# - Roberta model also got the same review wrong

# **Negative Sentiment Prediction but high star review**
# 
# Which are the examples that our models classifies as having a negative sentiment but it actually has a high star rating.

# In[32]:


results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['review_text'].values[0]


# - The sentence does contain the words like disappointment and user does want some changes in product but user has overall positive sentiment towards the product

# In[33]:


results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['review_text'].values[0]


# - In the text review the user expresses diagreement with change in product price, so our model classified it as a negative sentiment but user's overall sentiment towards the product is positive

# ## Directly predicting the star rating from the text review

# Using the **BERT** based model from hugging face 🤗 to classify the text between 1 to 5 star rating depending on the sentiment of the comment.
# - Instead of just predicting if the review is positive or not, we can get a more quantitative result in form of rating.

# In[34]:


#load the model
ratings_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
ratings_tokenizer = AutoTokenizer.from_pretrained(ratings_model_name)
ratings_model = AutoModelForSequenceClassification.from_pretrained(ratings_model_name)


# In[35]:


print("Example review:", example)
print("\nActual Rating:", df['Score'][50])


# In[36]:


ratings_encoded_text = ratings_tokenizer(example, return_tensors='pt')
ratings_encoded_text


# In[37]:


ratings_output = ratings_model(**ratings_encoded_text)


# In[38]:


ratings_output.logits[0].detach().numpy()


# In[39]:


#we have to add one as indexing starts from 0 but star ratings are from 1 to 5
ratings_pred = np.argmax(ratings_output.logits[0].detach().numpy()) + 1
print("Predicted Star rating:", ratings_pred)


# In[40]:


def rating_prediction_bert(text):
    ratings_encoded_text = ratings_tokenizer(text, return_tensors='pt')
    res = ratings_model(**ratings_encoded_text)
    rating = np.argmax(res.logits[0].detach().numpy()) + 1
    
    return {'rating' : rating}


# **Make prediction for all reviews**

# In[41]:


ratings_results = {}
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row['review_text']

    #BERT Model can only accept text with max length of 514
    if len(text) > 514:
        text = text[:514]

    #making rating predictions using bert model
    rating_prediction = rating_prediction_bert(text)

    ratings_results[idx] = rating_prediction


# In[42]:


ratings_df = pd.DataFrame(ratings_results).T
ratings_df = ratings_df.merge(df, how = 'left', right_index=True, left_index=True)
ratings_df.head()


# In[43]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(ratings_df['rating'], ratings_df['Score'])
accuracy


# In[44]:


one_off_accuracy = np.where(abs(ratings_df['rating'] - ratings_df['Score']) <= 1, 1, 0).sum() / ratings_df.shape[0]
one_off_accuracy


# - We can exactly predict the rating of review with ~70% accuracy and can predict the rating very closely (just one more or less) with 95% accuracy.
# 
# - So we can now say that model works considerably well and can be used for understanding the users sentiments over new products and services in the future or we can use it to analyze some other existing products or businesses.

# ## Using our Models to make predictions on new data scraped from Yelp to perform Sentiment Analysis 

# Now we will select a restauraunt business on Yelp and collect some reviews from the customers to see if this is a place where people like to eat or NOT, we can do a similar thing for any business and collect data from multiple sources also like twitter, reddit, youtube, facebook, instagram and more.

# In[45]:


import requests
from bs4 import BeautifulSoup
import re


# In[46]:


r = requests.get('https://www.yelp.com/biz/mejico-sydney-2')
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]


# In[47]:


reviews[:2]


# In[48]:


#store reviews in a dataframe
yelp_reviews = pd.DataFrame(np.array(reviews), columns=['review_text'])
yelp_reviews.head()


# In[55]:


#Now we can make perform some sentiment analysis over these text reviews
#using RoBERTa model to get different component scores (pos, neu and neg)
#and BERT to predict numerical star ratings
yelp_results = {}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row['review_text']

    #Roberta Model can only accept text with max length of 514
    if len(text) > 514:
        text = text[:514]

    #making predictions using roberta model
    roberta_result = polarity_scores_roberta(text)
    
    #making star prediction using bert model
    bert_result = rating_prediction_bert(text)
    #combine both results 
    result = roberta_result | bert_result

    yelp_results[idx] = result   


# In[56]:


yelp_results_df = pd.DataFrame(yelp_results).T
yelp_results_df = yelp_results_df.merge(yelp_reviews, how = 'left', right_index=True, left_index=True)
yelp_results_df.head()


# In[52]:


yelp_results_df.rating.mean()


# **Conclusion:**
# 
# So using text reviews we predicted that the restraunt has overall rating of approximatly 4, and we can confirm that our model can give us a good idea about sentiment towards any business with just the text reviews because the actual rating of the restraunt is 4.5 and we were able to closely predict that with just a small sample of reviews and this can get better by analyzing all the reviews.
# 

# ## How is all of this useful?

# We can use hundereds of thousands of online comments and text reviews posted by people on social media about any particular product, movie, service, business or Ad campaign to understand the overall feeling of public towards that. We can use this to better select and market our products and make data driven decisions.
# 
# This can help us deliver better services and increase profit margins among other things.

# ## Function for making predictions

# In[58]:


#function to perform sentiment analysis for any given text and also predict the numerical rating out of 5
def predict_sentiment_and_rating(text):
    #making sentiment predictions using roberta model (positive, neutral and negative)
    roberta_result = polarity_scores_roberta(text)
    
    #making star prediction using bert model (stars / score out of 5)
    bert_result = rating_prediction_bert(text)
    
    #combine both results 
    result = roberta_result | bert_result
    
    return result

# Interactive function for real-time analysis
def analyze_comment_real_time():
    print("\n--- Real-time Sentiment Analysis ---")
    print("Enter a comment to analyze (or type 'quit' to exit):")
    while True:
        user_input = input("\nYour Review: ")
        if user_input.lower() == 'quit':
            print("Exiting real-time analysis.")
            break
        
        if not user_input.strip():
            continue
            
        try:
            result = predict_sentiment_and_rating(user_input)
            print("-" * 30)
            print(f"Analysis Results:")
            print(f"  - Predicted Rating: {result['rating']} Stars")
            print(f"  - Sentiment Confidence (RoBERTa):")
            print(f"    Positive: {result['roberta_pos']:.4f}")
            print(f"    Neutral:  {result['roberta_neu']:.4f}")
            print(f"    Negative: {result['roberta_neg']:.4f}")
            print("-" * 30)
        except Exception as e:
            print(f"An error occurred: {e}")

# predict_sentiment_and_rating("Oh I love this place this is the most amazing thing ever. It great every time I go there.")

# predict_sentiment_and_rating("No I am not a fan of this, not recommended at all.")


# **We can use this function in future to make predictions on any reviews to analyze the sentiment of the user and learn thier opinion quantitatively**
# 
# It can also be used to deploy a simple web app that works for a single review at a time or a batch of reviews together

if __name__ == "__main__":
    # If you want to run the full analysis script first, keep the code above as is.
    # To jump straight to real-time analysis, you can call it here.
    analyze_comment_real_time()
