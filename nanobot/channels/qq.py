"""QQ channel implementation using botpy SDK."""

import asyncio
import inspect
import mimetypes
import os
import uuid
from pathlib import Path
from collections import deque
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import httpx

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import QQConfig

try:
    import botpy
    from botpy.message import C2CMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            super().__init__(intents=intents)

        async def on_ready(self):
            logger.info(f"QQ bot ready: {self.robot.name}")

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message)

        async def on_direct_message_create(self, message):
            await channel._on_message(message)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._bot_task: asyncio.Task | None = None
        self._media_dir = Path.home() / ".nanobot" / "media" / "qq"
        self._cos_missing_warned = False

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.secret:
            logger.error("QQ app_id and secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        self._bot_task = asyncio.create_task(self._run_bot())
        logger.info("QQ bot started (C2C private message)")

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.secret)
            except Exception as e:
                logger.warning(f"QQ bot error: {e}")
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._bot_task:
            self._bot_task.cancel()
            try:
                await self._bot_task
            except asyncio.CancelledError:
                pass
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return

        # For compatibility with existing QQ bot APIs, send normal text first,
        # then try media APIs best-effort per item.
        fallback_lines: list[str] = []
        try:
            content = (msg.content or "").strip()
            if content:
                await self._client.api.post_c2c_message(
                    openid=msg.chat_id,
                    msg_type=0,
                    content=content,
                )

            for media_item in msg.media or []:
                ok = await self._send_media_best_effort(openid=msg.chat_id, media_item=str(media_item))
                if not ok:
                    fallback_lines.append(str(media_item))

            if fallback_lines:
                tips = "\n".join(f"[media] {line}" for line in fallback_lines if line)
                if tips:
                    await self._client.api.post_c2c_message(
                        openid=msg.chat_id,
                        msg_type=0,
                        content=tips[:1800],
                    )
        except Exception as e:
            logger.error(f"Error sending QQ message: {e}")

    async def _send_media_best_effort(self, openid: str, media_item: str) -> bool:
        """Try botpy media APIs; return False if sending media is not possible."""
        if not self._client or not media_item:
            return False
        api = self._client.api
        path = Path(media_item).expanduser()
        is_local_file = path.exists() and path.is_file()
        is_url = self._looks_like_url(media_item)
        media_type = self._guess_media_type(media_item)

        media_url = ""
        if is_local_file and not is_url:
            media_url = await self._upload_local_to_cos(path)
            if not media_url:
                return False
        elif is_url:
            media_url = media_item

        if media_url:
            post_c2c_file = getattr(api, "post_c2c_file", None)
            if post_c2c_file:
                try:
                    media = await post_c2c_file(openid=openid, file_type=media_type, url=media_url, srv_send_msg=False)
                    # msg_type=7 follows QQ media message type convention.
                    await api.post_c2c_message(openid=openid, msg_type=7, media=media)
                    return True
                except Exception as e:
                    logger.debug(f"QQ media send via post_c2c_file failed: {e}")

            # Fallback: try raw media/url fields if running against a different botpy variant.
            for kwargs in (
                {"openid": openid, "msg_type": 7, "media": media_url},
                {"openid": openid, "msg_type": 0, "content": media_url},
            ):
                filtered = self._filter_kwargs(api.post_c2c_message, kwargs)
                try:
                    await api.post_c2c_message(**filtered)
                    return True
                except Exception as e:
                    logger.debug(f"QQ media send via post_c2c_message fallback failed: {e}")
        return False

    async def _upload_local_to_cos(self, path: Path) -> str:
        """Upload local media file to Tencent COS and return a presigned URL."""
        bucket = (self.config.cos_bucket or "").strip()
        region = (self.config.cos_region or "").strip()
        secret_id = os.environ.get("COS_SECRET_ID", "").strip()
        secret_key = os.environ.get("COS_SECRET_KEY", "").strip()
        if not bucket or not region or not secret_id or not secret_key:
            if not self._cos_missing_warned:
                logger.warning(
                    "QQ COS upload disabled: missing cosBucket/cosRegion or COS_SECRET_ID/COS_SECRET_KEY"
                )
                self._cos_missing_warned = True
            return ""
        return await asyncio.to_thread(self._upload_local_to_cos_sync, path, bucket, region, secret_id, secret_key)

    def _upload_local_to_cos_sync(
        self, path: Path, bucket: str, region: str, secret_id: str, secret_key: str
    ) -> str:
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except Exception:
            logger.error("qcloud_cos not installed. Install with: pip install cos-python-sdk-v5")
            return ""

        key = self._build_cos_key(path)
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        try:
            config = CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key, Scheme="https")
            client = CosS3Client(config)
            with open(path, "rb") as f:
                client.put_object(
                    Bucket=bucket,
                    Body=f,
                    Key=key,
                    ContentType=content_type,
                )
            return client.get_presigned_download_url(
                Bucket=bucket,
                Key=key,
                Expired=max(60, int(self.config.cos_url_expire_seconds)),
            )
        except Exception as e:
            logger.error(f"QQ COS upload failed: {e}")
            return ""

    def _build_cos_key(self, path: Path) -> str:
        prefix = (self.config.cos_prefix or "nanobot/qq").strip().strip("/")
        suffix = path.suffix.lower() or ".bin"
        return f"{prefix}/{uuid.uuid4().hex}{suffix}"

    async def _on_message(self, data: "C2CMessage") -> None:
        """Handle incoming message from QQ."""
        try:
            # Dedup by message ID
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            author = data.author
            user_id = str(getattr(author, "id", None) or getattr(author, "user_openid", "unknown"))
            content = (getattr(data, "content", "") or "").strip()

            media_items = self._extract_media_items(data)
            media_paths: list[str] = []
            content_parts = [content] if content else []
            if media_items:
                media_paths = await self._download_media_items(media_items)
                for path in media_paths:
                    content_parts.append(f"[qq-media: {path}]")
                if not media_paths:
                    for item in media_items:
                        content_parts.append(f"[qq-media: {item}]")

            if not content_parts:
                return

            await self._handle_message(
                sender_id=user_id,
                chat_id=user_id,
                content="\n".join(content_parts),
                media=media_paths,
                metadata={
                    "message_id": data.id,
                    "attachments": media_items,
                },
            )
        except Exception as e:
            logger.error(f"Error handling QQ message: {e}")

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        try:
            p = urlparse(value)
            return p.scheme in ("http", "https") and bool(p.netloc)
        except Exception:
            return False

    @staticmethod
    def _guess_media_type(value: str) -> int:
        lower = value.lower()
        image_ext = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")
        if lower.endswith(image_ext):
            return 1
        return 2

    @staticmethod
    def _filter_kwargs(method: object, kwargs: dict[str, object]) -> dict[str, object]:
        """Only pass kwargs accepted by the target method signature."""
        try:
            sig = inspect.signature(method)
            accepted = set(sig.parameters.keys())
            return {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            return kwargs

    def _extract_media_items(self, data: "C2CMessage") -> list[str]:
        """Extract media URLs from QQ message payload in a schema-tolerant way."""
        out: list[str] = []
        candidate_lists = []
        for field in ("attachments", "attachment", "media", "images", "files"):
            value = getattr(data, field, None)
            if value:
                candidate_lists.append(value)
        for candidate in candidate_lists:
            values = candidate if isinstance(candidate, list) else [candidate]
            for item in values:
                if isinstance(item, str):
                    if self._looks_like_url(item):
                        out.append(item)
                    continue
                if isinstance(item, dict):
                    for key in ("url", "download_url", "proxy_url"):
                        v = item.get(key)
                        if isinstance(v, str) and self._looks_like_url(v):
                            out.append(v)
                else:
                    for key in ("url", "download_url", "proxy_url"):
                        v = getattr(item, key, None)
                        if isinstance(v, str) and self._looks_like_url(v):
                            out.append(v)
        # dedup while preserving order
        return list(dict.fromkeys(out))

    async def _download_media_items(self, urls: list[str]) -> list[str]:
        """Download inbound media URLs to ~/.nanobot/media/qq and return local paths."""
        paths: list[str] = []
        if not urls:
            return paths
        try:
            self._media_dir.mkdir(parents=True, exist_ok=True)
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                for i, url in enumerate(urls):
                    try:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        suffix = Path(urlparse(url).path).suffix
                        if not suffix or suffix == ".bin":
                            ctype = (resp.headers.get("content-type") or "").split(";")[0].strip()
                            guessed = mimetypes.guess_extension(ctype) if ctype else None
                            suffix = guessed or suffix or ".bin"
                        file_path = self._media_dir / f"{int(asyncio.get_running_loop().time() * 1000)}_{i}{suffix}"
                        file_path.write_bytes(resp.content)
                        paths.append(str(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to download QQ media: {e}")
        except Exception as e:
            logger.warning(f"Failed to prepare QQ media directory: {e}")
        return paths
