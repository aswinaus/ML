# Databricks notebook source
import requests
import pandas as pd
from urllib.parse import urlparse
import zipfile
import io

# Set your parameters
tenant_id = "115ee48c-5146-4054-9f33-83e2bfe089fd"
client_id = "98031710-e46d-48df-968b-f052709fb9cf"
client_secret = ""
site_url = "https://eyaswin.sharepoint.com/sites/eysitetesting"

# Step 1: Get access token
token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
scope = "https://graph.microsoft.com/.default offline_access"
token_params = {
    "client_id": client_id,
    "client_secret": client_secret,
    "grant_type": "client_credentials",
    "scope": scope
}
token_resp = requests.post(token_url, data=token_params)
access_token = token_resp.json().get("access_token")

# Step 2: Get site ID
site_api_url = f"https://graph.microsoft.com/v1.0/sites/{site_url.replace('https://', '').replace('/', ':')}"
headers = {
    "Authorization": f"Bearer {access_token}",
    "Accept": "application/json"
}
site_id_resp = requests.get(site_api_url, headers=headers)
site_id = site_id_resp.json().get("id")

# Step 3: List document libraries (drives)
drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
drives_resp = requests.get(drives_url, headers=headers)
drives = drives_resp.json().get("value", [])

libraries = []
files = []
zip_file_details = []

for drive in drives:
    drive_id = drive.get("id")
    drive_name = drive.get("name")
    libraries.append({"library_id": drive_id, "library_name": drive_name})
    
    # List files in the root of each library
    items_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root/children"
    items_resp = requests.get(items_url, headers=headers)
    items = items_resp.json().get("value", [])
    for item in items:
        if not item.get("folder"):  # Only files, not folders
            files.append({
                "library_name": drive_name,
                "document_name": item.get("name")
            })
            # If file is a zip, get its contents
            if item.get("name", "").lower().endswith(".zip"):
                file_id = item.get("id")
                download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
                zip_resp = requests.get(download_url, headers=headers, stream=True)
                if zip_resp.status_code == 200:
                    try:
                        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as z:
                            zip_filenames = z.namelist()
                            for filename in zip_filenames:
                                zip_file_details.append({
                                    "library_name": drive_name,
                                    "zip_file_name": item.get("name"),
                                    "file_within_zip": filename
                                })
                    except zipfile.BadZipFile:
                        zip_file_details.append({
                            "library_name": drive_name,
                            "zip_file_name": item.get("name"),
                            "file_within_zip": "Corrupted or invalid zip file"
                        })

display(pd.DataFrame(libraries))
display(pd.DataFrame(files))
display(pd.DataFrame(zip_file_details))

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/documentlibraries/")

dbutils.fs.ls("/documentlibraries/")
