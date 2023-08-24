import searchtweets
import configparser

import bruh


'''
config = configparser.ConfigParser()
config.read('config.ini')



name = config.get('creds', 'name')
api_key = config.get('creds', 'api_key')
api_key_secret = config.get('creds', 'api_key_secret')
bearer_token = config.get('creds', 'bearer_token')
access_token = config.get('creds', 'access_token')
access_token_secret = config.get('creds', 'access_token_secret')
'''

name = bruh.name
api_key = bruh.api_key
api_key_secret = bruh.api_key_secret
bearer_token = bruh.bearer_token
access_token = bruh.access_token
access_token_secret = bruh.access_token_secret
zbruh = [name, api_key, api_key_secret, bearer_token, access_token, access_token_secret]

for b in zbruh:
    print(type(b))
    print(str(b))