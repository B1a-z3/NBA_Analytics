"""
stats.nba.com now fingerprints the TLS handshake itself (via Akamai bot
management), not just HTTP headers. Plain `requests`/`urllib3` (what nba_api
uses internally) doesn't match a real browser's TLS ClientHello, so calls
silently hang until timeout instead of returning an error -- confirmed by
direct testing: identical requests via `requests` time out after 15s on
every stats.nba.com endpoint tried (commonplayerinfo, scoreboardv2), while
the same request via `curl_cffi` with `impersonate="chrome"` (which
replicates a real Chrome TLS fingerprint) returns HTTP 200 in ~0.4s.

This module monkeypatches the `requests` reference inside
`nba_api.library.http` to route through curl_cffi instead. Import this
BEFORE importing anything from `nba_api.stats.endpoints` (both
pull_nba_api_live.py and enrich_player_bio.py do this).

If NBA ever changes their bot-mitigation approach and plain `requests`
works fine again, this module is a no-op risk -- worst case it's an
unnecessary dependency, not a correctness problem.
"""
import curl_cffi.requests as curl_requests
import nba_api.library.http as _nba_http_module


class _CurlCffiShim:
    """Minimal drop-in for the subset of `requests` that nba_api's
    NBAHTTP.send_api_request() actually calls: requests.get(url=..., params=...,
    headers=..., proxies=..., timeout=...) -> object with .url/.status_code/.text
    """

    @staticmethod
    def get(url, params=None, headers=None, proxies=None, timeout=None):
        kwargs = {"impersonate": "chrome"}
        if params is not None:
            kwargs["params"] = params
        if headers is not None:
            kwargs["headers"] = headers
        if proxies is not None:
            # curl_cffi expects a single proxy URL string, not a requests-style
            # {"http": ..., "https": ...} dict
            kwargs["proxy"] = proxies.get("https") or proxies.get("http")
        if timeout is not None:
            kwargs["timeout"] = timeout
        return curl_requests.get(url, **kwargs)


_nba_http_module.requests = _CurlCffiShim()
