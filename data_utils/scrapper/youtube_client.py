from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional
import re
from datetime import datetime, date

from googleapiclient.discovery import build


@dataclass
class VideoInfo:
    id: str
    title: str
    published_at: str


@dataclass
class Comment:
    video_id: str
    video_title: str
    comment_id: str
    text: str
    author: str
    like_count: int
    published_at: str
    updated_at: str
    parent_id: Optional[str] = None


class YouTubeClient:
    """Thin wrapper around the YouTube Data API for search + comments."""

    def __init__(self, api_key: str):
        self.service = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    def resolve_channel_id(self, channel_url: str) -> str:
        """Resolve a channel URL/handle to a channel ID. Expects handle-style URLs."""
        handle = self._extract_handle(channel_url)
        request = self.service.search().list(
            part="snippet", q=handle, type="channel", maxResults=1,
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            raise ValueError(f"Could not resolve channel from URL/handle: {channel_url!r}")
        return items[0]["id"]["channelId"]

    def iter_channel_videos(
        self,
        channel_id: str,
        max_results: int = -1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[VideoInfo]:
        """
        Yield videos from a channel (ordered by recency) until exhaustion or max_results.

        Uses the channel uploads playlist to include unlisted items (search API misses those).
        """
        playlist_id = self._uploads_playlist_id(channel_id)
        if not playlist_id:
            return

        start_dt = _parse_iso_date(start_date) if start_date else None
        end_dt = _parse_iso_date(end_date) if end_date else None

        next_page_token: Optional[str] = None
        fetched = 0

        while True:
            page_size = 50 if max_results is None or max_results < 0 else min(50, max_results - fetched)
            if page_size <= 0:
                break

            request = self.service.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=page_size,
                pageToken=next_page_token,
            )
            response = request.execute()

            items = response.get("items", [])
            for item in items:
                snippet = item.get("snippet", {})
                vid = snippet.get("resourceId", {}).get("videoId")
                if not vid:
                    continue
                published_at = snippet.get("publishedAt", "")
                pub_dt = _parse_iso_datetime(published_at)

                if end_dt and pub_dt and pub_dt.date() > end_dt:
                    # Too new; skip but continue (playlist ordered newest first).
                    continue
                if start_dt and pub_dt and pub_dt.date() < start_dt:
                    # Older than desired window; we can stop iterating.
                    return

                yield VideoInfo(
                    id=vid,
                    title=snippet.get("title", ""),
                    published_at=published_at,
                )
                fetched += 1
                if max_results is not None and max_results >= 0 and fetched >= max_results:
                    return

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    @staticmethod
    def _extract_handle(url: str) -> str:
        # Handles look like https://www.youtube.com/@handle
        match = re.search(r"@([^/?#]+)", url)
        if match:
            return match.group(1)
        raise ValueError("Channel URL must contain a handle (e.g., https://www.youtube.com/@handle)")

    def _uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """Return the uploads playlist ID for a channel."""
        resp = (
            self.service.channels()
            .list(part="contentDetails", id=channel_id, maxResults=1)
            .execute()
        )
        items = resp.get("items", [])
        if not items:
            return None
        return items[0].get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")

    def fetch_comments(
        self, video_id: str, video_title: str, max_comments: int = -1, include_replies: bool = True
    ) -> List[Comment]:
        """Fetch comment threads (optionally including replies) up to max_comments (-1 = all)."""
        comments: List[Comment] = []
        next_page_token: Optional[str] = None

        while True:
            request = self.service.commentThreads().list(
                part="snippet,replies" if include_replies else "snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText",
            )
            response = request.execute()
            for item in response.get("items", []):
                thread = item["snippet"]["topLevelComment"]
                comments.append(self._extract_comment(thread, video_id, video_title))

                if include_replies:
                    for reply in item.get("replies", {}).get("comments", []):
                        comments.append(
                            self._extract_comment(
                                reply, video_id, video_title, parent_id=thread["id"]
                            )
                        )

                if max_comments is not None and max_comments >= 0 and len(comments) >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if (max_comments is not None and max_comments >= 0 and len(comments) >= max_comments) or (
                not next_page_token
            ):
                break

        return comments

    @staticmethod
    def _extract_comment(
        resource: Dict, video_id: str, video_title: str, parent_id: Optional[str] = None
    ) -> Comment:
        snippet = resource.get("snippet", {})
        return Comment(
            video_id=video_id,
            video_title=video_title,
            comment_id=resource.get("id", ""),
            text=snippet.get("textDisplay", ""),
            author=snippet.get("authorDisplayName", ""),
            like_count=int(snippet.get("likeCount", 0) or 0),
            published_at=snippet.get("publishedAt", ""),
            updated_at=snippet.get("updatedAt", ""),
            parent_id=parent_id,
        )


def _parse_iso_datetime(value: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _parse_iso_date(value: str) -> date:
    return datetime.fromisoformat(value).date()
