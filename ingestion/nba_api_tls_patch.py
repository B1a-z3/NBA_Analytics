"""
stats.nba.com now fingerprints the TLS handshake itself (via Akamai bot
management), not just HTTP headers. Plain `requests`/`urllib3` (what nba_api
uses internally) doesn't match a real browser's TLS ClientHello, so calls
silently hang until timeout instead of returning an error -- confirmed by
direct testing: identical requests via `requests` time out after 15s on
every stats.nba.com endpoint tried (commonplayerinfo, scoreboardv2), while
the same request via `curl_cffi` with `impersonate="chrome"` (which
replicates a real Chrome TLS fingerprint) returns HTTP 200 in ~0.3s.

This module monkeypatches the `requests` reference inside
`nba_api.library.http` to route through curl_cffi instead. Import this
BEFORE importing anything from `nba_api.stats.endpoints` (both
pull_nba_api_live.py and enrich_player_bio.py do this).

Note on nba_api's internal HTTP mechanism (this has changed between
versions, and broke this patch silently once already -- see below):
  - nba_api 1.4.1: NBAHTTP.send_api_request() calls the module-level
    `requests.get(...)` directly.
  - nba_api 1.11.4+: it calls `self.get_session().get(...)`, where
    get_session() lazily creates and CACHES a `requests.Session()` as a
    class attribute (NBAHTTP._session). A patch that only replaces
    `requests.get` (the old approach) is a silent no-op here -- the real
    HTTP call happens through a Session instance instead, so it still
    goes out via plain requests/urllib3 and hangs exactly as if unpatched.
This patch covers BOTH mechanisms so it keeps working across nba_api
upgrades: it replaces the whole `requests` reference with a shim exposing
both `.get()` (old path) and `.Session` (new path, backed by curl_cffi's
own Session with impersonate="chrome" pre-set).

If NBA ever changes their bot-mitigation approach and plain `requests`
works fine again, this module is a no-op risk -- worst case it's an
unnecessary dependency, not a correctness problem.
"""
import curl_cffi.requests as curl_requests
import nba_api.library.http as _nba_http_module


class _ImpersonatingSession(curl_requests.Session):
    """curl_cffi Session that always impersonates Chrome's TLS fingerprint,
    so nba_api's `self.get_session()` -> `Session()` -> `.get(...)` chain
    (nba_api 1.11.4+) gets the fingerprint without nba_api itself knowing
    curl_cffi exists."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("impersonate", "chrome")
        super().__init__(*args, **kwargs)


class _CurlCffiShim:
    """Drop-in for the subset of the `requests` module nba_api's
    NBAHTTP.send_api_request() calls, across both mechanisms above."""

    Session = _ImpersonatingSession

    @staticmethod
    def get(url, params=None, headers=None, proxies=None, timeout=None):
        kwargs = {"impersonate": "chrome"}
        if params is not None:
            kwargs["params"] = params
        if headers is not None:
            kwargs["headers"] = headers
        if proxies is not None:
            # curl_cffi's top-level `get()` expects a single proxy URL
            # string, not a requests-style {"http": ..., "https": ...} dict
            kwargs["proxy"] = proxies.get("https") or proxies.get("http")
        if timeout is not None:
            kwargs["timeout"] = timeout
        return curl_requests.get(url, **kwargs)


_nba_http_module.requests = _CurlCffiShim()

# nba_api 1.11.4+ caches its Session as a class attribute the first time
# get_session() runs. If anything imported nba_api.stats.endpoints before
# this patch applied in the same process, that cache could already hold a
# real (unpatched) requests.Session -- clear it so the next call is forced
# to create a fresh one via the shim above.
if hasattr(_nba_http_module.NBAHTTP, "_session"):
    _nba_http_module.NBAHTTP._session = None
