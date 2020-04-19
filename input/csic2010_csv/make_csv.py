import pandas as pd
import numpy as np

#df = pd.read_csv("test.csv")
df = pd.read_csv("output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv")

new_dict = {}

test = df.groupby('index').agg({'payload': list})
#print(test)
#for index, row in test.iterrows():
#    print(type(row['payload']))
#    print(row['payload'])

for index, row in df.iterrows():
    if row['index'] in new_dict:
        pass
    else:
        new_dict[row['index']] = 1
        if row['method'] == 'GET':
            raw_text = "{0} {1}{2} {3}\n".format(row['method'], row['url'], ("?"+"&".join(test.at[int(row['index']), 'payload'])) if np.nan not in test.at[int(row['index']), 'payload'] else "", row['protocol'])
            raw_text += "User-Agent: {0}\n".format(row['userAgent'])
            raw_text += "Pragma: {0}\n".format(row['pragma'])
            raw_text += "Cache-control: {0}\n".format(row['cacheControl'])
            raw_text += "Accept: {0}\n".format(row['accept'])
            raw_text += "Accept-Encoding: {0}\n".format(row['acceptEncoding'])
            raw_text += "Accept-Charset: {0}\n".format(row['acceptCharset'])
            raw_text += "Accept-Language: {0}\n".format(row['acceptLanguage'])
            raw_text += "Host: {0}\n".format(row['host'])
            raw_text += "Cookie: {0}\n".format(row['cookie'])
            raw_text += "Connection: {0}\n".format(row['connection'])
        elif row['method'] == 'POST':
            raw_text = "{0} {1} {2}\n".format(row['method'], row['url'], row['protocol'])
            raw_text += "User-Agent: {0}\n".format(row['userAgent'])
            raw_text += "Pragma: {0}\n".format(row['pragma'])
            raw_text += "Cache-control: {0}\n".format(row['cacheControl'])
            raw_text += "Accept: {0}\n".format(row['accept'])
            raw_text += "Accept-Encoding: {0}\n".format(row['acceptEncoding'])
            raw_text += "Accept-Charset: {0}\n".format(row['acceptCharset'])
            raw_text += "Accept-Language: {0}\n".format(row['acceptLanguage'])
            raw_text += "Host: {0}\n".format(row['host'])
            raw_text += "Cookie: {0}\n".format(row['cookie'])
            raw_text += "Content-Type: {0}\n".format(row['contentType'])
            raw_text += "Connection: {0}\n".format(row['connection'])
            raw_text += "Content-Length: {0}\n".format(row['contentLength'])
            raw_text += "\n{0}".format(test[row['index']])
        print(raw_text)
