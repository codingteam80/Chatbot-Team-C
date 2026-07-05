import re
import uuid

BROWSER_ID_SESSION_KEY = "browser_id"
BROWSER_ID_QUERY_KEY = "bid"
BROWSER_ID_SYNC_SESSION_KEY = "browser_id_query_synced"
BROWSER_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{12,80}$")


def is_valid_browser_id(value):
    # Accept only safe browser IDs.
    return bool(value and BROWSER_ID_RE.match(str(value)))


def get_query_browser_id(st):
    # Read the browser ID from URL query params.
    try:
        value = st.query_params.get(BROWSER_ID_QUERY_KEY)
    except Exception:
        try:
            value = st.experimental_get_query_params().get(BROWSER_ID_QUERY_KEY, [None])
        except Exception:
            value = None

    if isinstance(value, list):
        value = value[0] if value else None

    return str(value) if is_valid_browser_id(value) else None


def set_query_browser_id(st, browser_id):
    # Write it to the URL only when needed.
    # Updating query params can trigger another rerun, so avoid duplicate writes.
    if not is_valid_browser_id(browser_id):
        return

    if get_query_browser_id(st) == browser_id:
        st.session_state[BROWSER_ID_SYNC_SESSION_KEY] = True
        return

    if st.session_state.get(BROWSER_ID_SYNC_SESSION_KEY):
        return

    try:
        st.query_params[BROWSER_ID_QUERY_KEY] = browser_id
        st.session_state[BROWSER_ID_SYNC_SESSION_KEY] = True
        return
    except Exception:
        pass

    try:
        st.experimental_set_query_params(**{BROWSER_ID_QUERY_KEY: browser_id})
        st.session_state[BROWSER_ID_SYNC_SESSION_KEY] = True
    except Exception:
        pass


def get_browser_id(st):
    # Create or get a browser ID. This is not authentication.
    existing_id = st.session_state.get(BROWSER_ID_SESSION_KEY)

    if is_valid_browser_id(existing_id):
        return existing_id

    browser_id = get_query_browser_id(st) or uuid.uuid4().hex
    st.session_state[BROWSER_ID_SESSION_KEY] = browser_id
    set_query_browser_id(st, browser_id)

    return browser_id
