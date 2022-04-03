import base64
import pandas as pd


def create_onedrive_directdownload (onedrive_link):
    data_bytes64 = base64.b64encode(bytes(onedrive_link, 'utf-8'))
    data_bytes64_String = data_bytes64.decode('utf-8').replace('/','_').replace('+','-').rstrip("=")
    resultUrl = f"https://api.onedrive.com/v1.0/shares/u!{data_bytes64_String}/root/content"
    return resultUrl


onedrive_link = "https://1drv.ms/x/s!AsyQPQRa2P2OjHWxtDsJ4Jl-06ng?e=9yX680"
onedrive_direc_link = create_onedrive_directdownload(onedrive_link)

df = pd.read_excel(onedrive_direc_link)
print(df.head())
