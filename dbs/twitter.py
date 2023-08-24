import searchtweets
import configparser

config = configparser.ConfigParser()
config.read('config.ini')



name = config.get('creds', 'name')
api_key = config.get('creds', 'api_key')
api_key_secret = config.get('creds', 'api_key_secret')
bearer_token = config.get('creds', 'bearer_token')
access_token = config.get('creds', 'access_token')
access_token_secret = config.get('creds', 'access_token_secret')

bruh = [name, api_key, api_key_secret, bearer_token, access_token, access_token_secret]

for b in bruh:
    print(type(b))
    print(str(b))