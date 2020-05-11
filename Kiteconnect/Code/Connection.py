# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:58:40 2018

@author: sunilbhandari1
"""

#from Code.ActualExecution import Exec
from kiteconnect import KiteConnect
api_key = "d5qoavxjuaw18al8"
api_secret = "b0b39jlk7sg23mr91247snu5e742j0hn"
kite = KiteConnect(api_key=api_key)
kite.login_url()

data = kite.generate_session("35HUAXm8l02r5Ok0YlvtbyaF9tlt5P0u", api_secret=api_secret)
kite.set_access_token(data["access_token"])
print(kite.profile())

# Exec(kite)