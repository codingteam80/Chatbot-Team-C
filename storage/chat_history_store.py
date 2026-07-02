"""SQLite persistent chat history scoped by browser id."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

try:
    from config.settings import STORAGE_DIR, APP_TIMEZONE
except Exception:
    STORAGE_DIR = "storage"
    APP_TIMEZONE = "Asia/Manila"

DB_DIR = Path(STORAGE_DIR)
DB_PATH = DB_DIR / "chat_history.sqlite3"
DEFAULT_HISTORY_LIMIT = 30
GLOBAL_BROWSER_ID = "global"
TITLE_CHAR_LIMIT = 60
_DB_INITIALIZED = False


def get_now_text():
    # Timestamp para sa DB at message bubbles.
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(APP_TIMEZONE)).strftime("%Y-%m-%d %H:%M:%S")

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_connection():
    # Short-lived SQLite connection.
    DB_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def get_table_columns(connection, table_name):
    # Kunin ang existing columns ng table.
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def migrate_database(connection):
    # Safe migrations para sa older DB files.
    conversation_columns = get_table_columns(connection, "conversations")
    message_columns = get_table_columns(connection, "messages")

    if "browser_id" not in conversation_columns:
        connection.execute(
            "ALTER TABLE conversations ADD COLUMN browser_id TEXT NOT NULL DEFAULT 'global'"
        )

    if "created_at" not in message_columns:
        connection.execute(
            "ALTER TABLE messages ADD COLUMN created_at TEXT NOT NULL DEFAULT ''"
        )

    connection.execute(
        """
        UPDATE messages
        SET created_at = ?
        WHERE created_at IS NULL OR created_at = ''
        """,
        (get_now_text(),),
    )


def init_history_db():
    # Gumawa o i-migrate ang chat history DB once per Python process.
    global _DB_INITIALIZED

    if _DB_INITIALIZED and DB_PATH.exists():
        return

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                browser_id TEXT NOT NULL DEFAULT 'global',
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                question TEXT,
                sources_json TEXT,
                suggestions_json TEXT,
                sort_order INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
            """
        )
        migrate_database(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_browser_updated
            ON conversations(browser_id, updated_at DESC, id DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_order
            ON messages(conversation_id, sort_order, id)
            """
        )

    _DB_INITIALIZED = True


def clean_browser_id(browser_id):
    # Fallback kapag walang browser id.
    browser_id = str(browser_id or "").strip()
    return browser_id or GLOBAL_BROWSER_ID


def clean_title(text, fallback="New chat"):
    # Gawing compact ang conversation title.
    title = " ".join(str(text or "").split()) or fallback

    if len(title) > TITLE_CHAR_LIMIT:
        title = title[:TITLE_CHAR_LIMIT].rstrip() + "..."

    return title


def encode_json(value):
    # Safe JSON for SQLite.
    return json.dumps(value or [], ensure_ascii=False, default=str)


def decode_json(value, fallback=None):
    # Safe JSON decode.
    fallback = [] if fallback is None else fallback

    if not value:
        return fallback

    try:
        return json.loads(value)
    except Exception:
        return fallback


def conversation_belongs_to_browser(connection, conversation_id, browser_id):
    # I-check kung owned ng current browser ang conversation.
    if not conversation_id:
        return False

    row = connection.execute(
        """
        SELECT id
        FROM conversations
        WHERE id = ? AND browser_id = ?
        """,
        (conversation_id, clean_browser_id(browser_id)),
    ).fetchone()

    return row is not None


def create_conversation(browser_id, title=None):
    # Gumawa ng bagong conversation at ibalik ang id.
    init_history_db()
    now = get_now_text()
    browser_id = clean_browser_id(browser_id)

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO conversations (browser_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (browser_id, clean_title(title), now, now),
        )
        return cursor.lastrowid


def list_conversations(browser_id, limit=DEFAULT_HISTORY_LIMIT):
    # Recent conversations for current browser only.
    init_history_db()
    browser_id = clean_browser_id(browser_id)

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.browser_id = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (browser_id, int(limit)),
        ).fetchall()

    return [dict(row) for row in rows]


def load_messages(browser_id, conversation_id):
    # Load messages only if conversation belongs to this browser.
    if not conversation_id:
        return []

    init_history_db()
    browser_id = clean_browser_id(browser_id)

    with get_connection() as connection:
        if not conversation_belongs_to_browser(connection, conversation_id, browser_id):
            return []

        rows = connection.execute(
            """
            SELECT role, content, question, sources_json, suggestions_json, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY sort_order ASC, id ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:
        message = {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }

        if row["role"] == "assistant":
            message.update({
                "question": row["question"],
                "sources": decode_json(row["sources_json"], fallback=[]),
                "suggestions": decode_json(row["suggestions_json"], fallback=[]),
            })

        messages.append(message)

    return messages


def replace_conversation_messages(browser_id, conversation_id, messages, title=None):
    # Replace all messages para simple ang save after regenerate/edit.
    init_history_db()
    browser_id = clean_browser_id(browser_id)
    now = get_now_text()

    with get_connection() as connection:
        if not conversation_belongs_to_browser(connection, conversation_id, browser_id):
            cursor = connection.execute(
                """
                INSERT INTO conversations (browser_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (browser_id, clean_title(title), now, now),
            )
            conversation_id = cursor.lastrowid
        else:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?, updated_at = ?
                WHERE id = ? AND browser_id = ?
                """,
                (clean_title(title), now, conversation_id, browser_id),
            )

        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))

        for sort_order, message in enumerate(messages or []):
            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()

            if role not in {"user", "assistant"} or not content:
                continue

            connection.execute(
                """
                INSERT INTO messages (
                    conversation_id,
                    role,
                    content,
                    question,
                    sources_json,
                    suggestions_json,
                    sort_order,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    message.get("question"),
                    encode_json(message.get("sources", [])),
                    encode_json(message.get("suggestions", [])),
                    sort_order,
                    str(message.get("created_at") or now),
                ),
            )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE id = ? AND browser_id = ?
            """,
            (now, conversation_id, browser_id),
        )

    return conversation_id


def delete_conversation(browser_id, conversation_id):
    # Delete only one saved conversation for the current browser.
    # This is used by Clear Chat so it does not wipe the full sidebar history.
    if not conversation_id:
        return False

    init_history_db()
    browser_id = clean_browser_id(browser_id)

    with get_connection() as connection:
        if not conversation_belongs_to_browser(connection, conversation_id, browser_id):
            return False

        # Explicitly delete messages first for older DB files, then delete the conversation row.
        # The FK cascade still protects the normal path, but this keeps the cleanup predictable.
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        cursor = connection.execute(
            "DELETE FROM conversations WHERE id = ? AND browser_id = ?",
            (conversation_id, browser_id),
        )

    return cursor.rowcount > 0


def delete_all_conversations(browser_id):
    # Clear all saved chats for current browser only.
    init_history_db()

    with get_connection() as connection:
        connection.execute(
            "DELETE FROM conversations WHERE browser_id = ?",
            (clean_browser_id(browser_id),),
        )
