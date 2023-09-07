import searchtweets
from searchtweets import ResultStream, collect_results, load_credentials
import bruh
#secret file 

#SECRETS
name = bruh.name
api_key = bruh.api_key
api_key_secret = bruh.api_key_secret
bearer_token = bruh.bearer_token
access_token = bruh.access_token
access_token_secret = bruh.access_token_secret
zbruh = [name, api_key, api_key_secret, bearer_token, access_token, access_token_secret]

p_search_args = load_credentials("/.twitter_keys.yaml", yaml_key="search_tweets_api")


search_args = {
    "bearer_token": bearer_token,
}

query = "dog"
max_results = 100


# Perform the historical search and collect results
#ztweets = collect_results(query=query,
#                          max_results=500,
#                          result_stream_args=search_args,
#                          from_date="2022-01-01",
#                          to_date="2022-01-02")

tweets = collect_results(query=query, max_results=max_results, result_stream_args=p_search_args)

# Count occurrences of the keyword
keyword_count = sum(1 for tweet in tweets if "dog" in tweet.all_text.lower())

print(f"The keyword 'dog' was mentioned {keyword_count} times in the search results.")
