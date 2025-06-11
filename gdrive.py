#!/usr/bin/env python3
import argparse
import io
from typing import Any

from pathlib import Path
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from rich.console import Console
from rich.table import Table
from rich.pretty import pprint
from google.auth.exceptions import RefreshError
from rich.text import Text

console = Console()

# https://developers.google.com/drive/api/guides/about-sdk

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive"]

# created by following https://developers.google.com/drive/api/quickstart/python
Credentials_JSON = (
    Path.home() / "client_secret_32374387863-e7jikh2ktb31m25l27liebbunt0nfnkv.apps.googleusercontent.com.json"
)

TOKEN_FILE = Path.home() / "Downloads" / "token.json"

mime_types = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".7z": "application/x-compressed",
    ".zip": "application/zip",
    ".csv": "text/plain",
    ".txt": "text/plain",
}


def print_table(items: list[dict[str, Any]]) -> None:
    # items.sort(key=lambda e: e["name"])
    table = Table(title="Files")
    table.add_column("Name", style="cyan", no_wrap=True, max_width=32)
    table.add_column("Size (kb)", justify="right", style="cyan")
    table.add_column("Modified", style="yellow")
    table.add_column("Type", style="cyan")
    table.add_column("ID", style="cyan", no_wrap=True, max_width=10)

    for item in items:
        dt = datetime.strptime(item["modifiedTime"], "%Y-%m-%dT%H:%M:%S.%fZ")
        sz = round(int(item.get("size", 0)) / 1024)
        table.add_row(
            Text(item["name"], style="red" if item["trashed"] else "cyan"),  # Inline style
            str(sz),
            str(dt.strftime("%Y-%m-%d %H:%M")),
            item["mimeType"],
            item["id"],
        )

    console.print(table)


def login(token_file: Path) -> Credentials | None:
    creds = None
    # The file TOKEN_FILE stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # If refresh fails, force a new login
                creds = None
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(Credentials_JSON, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return creds


def download_file(drive_service: Resource, file_id: str, fn: Path) -> None:
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        console.print(f"Download {int(status.progress() * 100)}%")

    with open(fn, "wb") as f:
        f.write(fh.getvalue())
    console.print(f"Downloaded file {file_id} to {fn}", style="green")


def create_file(drive_service: Resource, fn: Path) -> str | None:
    """Insert new file.
    Returns : Id's of the file uploaded

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """

    try:
        file_metadata = {"name": fn.name}
        media = MediaFileUpload(fn, mimetype=mime_types[fn.suffix])
        # pylint: disable=maybe-no-member
        file = drive_service.files().create(body=file_metadata, media_body=media, fields="id").execute()
        console.print(f"File created {fn} with Google Drive id = {file.get('id')}", style="green")
        return file.get("id")

    except HttpError as error:
        console.print(f"An error occurred: {error}", style="red")

    return None


def update_file(drive_service: Resource, file_id: str, fn: Path) -> str | None:
    try:
        # file_metadata = {"id": file_id}
        media = MediaFileUpload(fn, mimetype=mime_types[fn.suffix])
        # pylint: disable=maybe-no-member
        file = drive_service.files().update(media_body=media, fileId=file_id, fields="id").execute()
        console.print(f"File updated {fn} with Google Drive id = {file.get('id')}", style="green")
        return file.get("id")

    except HttpError as error:
        console.print(f"An error occurred: {error}", style="red")

    return None


def find_file(drive_service: Resource, spec:str) -> str | None:
    """return file_id of highest version, not trashed file matching spec"""
    results = (
        drive_service.files()
        .list(q=f"name contains '{spec}'", fields="files(id, name, kind, version, trashed)")
        .execute()
    )
    existing = results.get("files", [])

    if existing:
        existing.sort(key=lambda e: int(e.get("version", 0)), reverse=True)
        for f in (e["id"] for e in existing if not e["trashed"]):
            return f
    else:
        console.print(f"File {spec} not found", style="yellow")

    return None


def list_files(drive_service: Resource) -> None:
    """
    Prints the names and ids of the first 16 files the user has access to.
    """
    results = (
        drive_service.files()
        .list(
            pageSize=16,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, version, trashed)",
            orderBy="modifiedTime desc",
        )
        .execute()
    )
    items = results.get("files", [])

    if not items:
        console.print("No files found.", style="red")
        return
    print_table(items)


def delete_file(drive_service: Resource, file_id: str) -> None:
    """Move a file to trash given its file_id."""
    try:
        # Update the file's trashed status
        drive_service.files().update(fileId=file_id, body={"trashed": True}).execute()
        console.print(f"File with ID: {file_id} has been moved to trash.", style="green")
    except Exception as e:
        console.print(f"Error deleting file: {e}", style="red")


def main(cmd: str, fname: str):
    try:
        if creds := login(TOKEN_FILE):
            service = build("drive", "v3", credentials=creds)
        match cmd:
            case "list":
                list_files(service)
                
            case "find" if fname:
                find_file(service, fname)
                
            case "up" if fname:
                fn = Path.home() / "Downloads" / fname
                if fn.exists():
                    if file_id := find_file(service, fname):
                        update_file(service, file_id, fn)
                    else:
                        create_file(service, fn)
                else:
                    console.print(f"File not found: {fn}", style="red")
                    
            case "dn" if fname:
                if file_id := find_file(service, fname):
                    fn = Path.home() / "Downloads" / fname
                    download_file(service, file_id, fn)
                    
            case "rm" if fname:
                if file_id := find_file(service, fname):
                    delete_file(service, file_id)
                else:
                    console.print(f"File not found: {fname}", style="red")
                    
            case _:
                console.print(f"Invalid command or missing filename: {cmd} {fname}", style="red")

    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        console.print(f"An error occurred: {error}", style="red")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Google Drive command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list            # List files in your Google Drive
  %(prog)s find document   # Find files containing 'document' in the name
  %(prog)s up report.xlsx  # Upload report.xlsx from Downloads folder
  %(prog)s dn report.xlsx  # Download report.xlsx to Downloads folder
  %(prog)s rm old_file.txt # Move old_file.txt to trash
        """)
    parser.add_argument(
        "cmd", choices=["list", "find", "up", "dn", "rm"], type=str, help="command [list|find|upload|download|rm]"
    )
    parser.add_argument("fn", type=str, nargs="?", default="", help="file name")
    args = parser.parse_args()
    main(args.cmd, args.fn)
