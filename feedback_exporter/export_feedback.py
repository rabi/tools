"""Feedback export module."""
import os
import json
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

def _fetch_feedback_data(database_url, app_base_url):
    parsed = urlparse(database_url)
    conn = psycopg2.connect(
        dbname=parsed.path.lstrip('/'),
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port
    )

    query = """
    SELECT
      f."value" AS score,
      CONCAT(%s, s."threadId") AS thread_url,
      s."input" AS step_input,
      child."output" AS step_output,
      f."comment",
      u."identifier" AS user_name,
      t."metadata" AS settings
    FROM
      "Feedback" f
    JOIN
      "Step" s ON f."stepId" = s."id"
    JOIN
      "Step" child ON child."parentId" = s."id"
    JOIN
      "Thread" t ON s."threadId" = t."id"
    JOIN
      "User" u ON t."userId" = u."id";
    """

    df = pd.read_sql_query(query, conn, params=(app_base_url,))
    df["settings"] = df["settings"].apply(lambda value: value["settings"])

    conn.close()
    return df

def _write_to_google_sheet(df, spreadsheet_id, credentials_json):
    creds_dict = json.loads(credentials_json)
    credentials = Credentials.from_service_account_info(creds_dict)
    service = build('sheets', 'v4', credentials=credentials)

    sheet_data = [df.columns.tolist()] + df.astype(str).values.tolist()
    body = {'values': sheet_data}

    # pylint: disable=no-member
    service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range='A1',
        valueInputOption='RAW',
        body=body
    ).execute()
    # pylint: enable=no-member

def main():
    """Entry point for feedback exporter."""
    database_url = os.getenv("DATABASE_URL")
    app_base_url = os.getenv("APP_BASE_URL", "http://app.com/thread/")
    spreadsheet_id = os.getenv("GOOGLE_SPREADSHEET_ID")
    credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")

    if not all([database_url, spreadsheet_id, credentials_json]):
        raise EnvironmentError("Missing one or more required environment variables.")

    df = _fetch_feedback_data(database_url, app_base_url)
    _write_to_google_sheet(df, spreadsheet_id, credentials_json)

if __name__ == "__main__":
    main()
