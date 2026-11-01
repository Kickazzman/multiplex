# REDACTION-LOG.md

Markers: `[REDACTED-IP]` (network addresses), `[REDACTED-PERSONAL]`
(emails, phone numbers), `[REDACTED-SECRET]` (credential-like tokens,
private-key material, password values). Only pattern kinds, counts, and
line numbers are logged here — never matched values.
Loopback addresses (127.x) and 0.0.0.0 are retained: they identify
nothing. The sweep pattern excludes them on the same basis.
Also retained, deliberately, and named here so the sweep's zero-hit
result is not misread: HH:MM:SS timestamps and DOI fragments are NOT
addresses (the patterns require valid octets, whole-quad boundaries,
and 3+ colon groups / '::' for IPv6, so neither matches); and the
documented test-fixture/example quads 192.168.0.1, 10.0.0.1,
255.255.255.255, 1.2.3.4, 8.8.8.8, 1.1.1.1 are kept where they are
spec values, because redacting them would corrupt trial specs whose
values ARE the data. Every other dotted quad / address is redacted.
Phone numbers match only with '-' / '.' separators or parentheses,
so bare digit sequences (PIDs, counters) are not touched.

## `hal_dispatch.py`
- ipv4: 2 occurrence(s), line(s): 28, 29

## `multiplex.py`
- ipv4: 1 occurrence(s), line(s): 61

## `profiles.py`
- ipv4: 2 occurrence(s), line(s): 46, 48

## `test_kyutai.py`
- ipv4: 1 occurrence(s), line(s): 41
