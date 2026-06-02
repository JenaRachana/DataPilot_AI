"""
Portable TLS trust bootstrap.

Tells Python to verify HTTPS certificates against the *operating system* trust
store (via the `truststore` package) instead of only the bundled `certifi` CAs.

Why this is portable and safe to keep on any machine:
  * On a corporate network that does TLS interception, the proxy's root CA is
    installed in the OS trust store (but not in certifi) — so this is what makes
    HTTPS work there.
  * On a personal/open network it simply uses the same public CAs the OS already
    trusts, so it behaves identically to the default.
  * Certificate verification stays ON in all cases — this never disables SSL.

If `truststore` is not installed for any reason, this is a no-op and the app
falls back to the default certifi behaviour.

Call install() once, as early as possible, before any HTTPS client is created.
"""


def install() -> None:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        # truststore unavailable -> fall back to default certifi trust store.
        pass
