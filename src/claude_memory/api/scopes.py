"""What an API key with scope "external" may do, and how its writes are marked.

Three changes of the Muse memory connector design (2026-09-16) read this module:
the endpoint allowlist the auth layer enforces (change 2), the curated
``/muse/openapi.json`` handed to Muse's connector builder (change 3), and the origin tag
stamped on every external write (change 4). They share one constant so the document Muse
is given and the rule the API applies cannot drift apart.

Nothing here imports FastAPI or the auth module: it is policy data plus a few string
functions, so the wiring lives in ``app.py`` and this stays cheap to import and to test.
"""

import unicodedata

#: The curated document's path. Named here rather than written twice, because the route
#: and the public-operation entry below have to stay in step.
MUSE_SPEC_PATH = "/muse/openapi.json"

#: Operations an "external" key may call: recall, list, get, store, update and delete,
#: link creation, the two vocabulary endpoints, and the key self-test.
#:
#: Each entry is (METHOD, path) where path is the route template exactly as FastAPI
#: records it on the APIRoute, which is byte-identical to the key under ``paths`` in the
#: OpenAPI document — so enforcement and spec generation share this set with no
#: translation step between them.
#:
#: Enforcement denies anything absent, which is what makes a privileged route added later
#: closed to external keys without anyone remembering to close it.
#:
#: ``GET /api/categories`` returns the categories IN USE across all users (a
#: ``SELECT DISTINCT`` over the rows), which exposes nothing recall does not already
#: expose and is worth having. It is NOT the list of legal values: a canonical category
#: with no live rows is absent from it, and it is not how an external client discovers
#: the vocabulary. That job belongs to the ``enum`` every category field carries in the
#: OpenAPI document (``models.CATEGORY_ENUM``), which lists the canonical set itself.
#:
#: ``GET /api/auth-check`` is granted so the key can be self-tested from Muse's side. It
#: answers with the caller's own user id and scope, which is the only way to see from
#: outside that a key meant to be external was written in API_KEYS' flat shape and landed
#: as a silent admin key. It reads nothing and exposes no other user.
#:
#: Link DELETE is deliberately absent. The design grants link creation, so an external
#: key can create a link it cannot remove through its own connector.
EXTERNAL_ALLOWED_OPERATIONS: frozenset[tuple[str, str]] = frozenset({
    ("POST", "/api/memories"),
    ("POST", "/api/memories/recall"),
    ("GET", "/api/memories"),
    ("GET", "/api/memories/{memory_id}"),
    ("PUT", "/api/memories/{memory_id}"),
    ("DELETE", "/api/memories/{memory_id}"),
    ("POST", "/api/memories/{memory_id}/links"),
    ("GET", "/api/tags"),
    ("GET", "/api/categories"),
    ("GET", "/api/auth-check"),
})

#: Operations that already serve the same bytes to an anonymous caller, so refusing a
#: scoped key on them would protect nothing and would break two real flows: a connector
#: builder that attaches its configured token to every request to the host, including the
#: spec fetch, and a human checking a freshly minted key with curl against /health.
#:
#: Kept apart from the allowlist rather than folded into it, so the curated document stays
#: exactly the allowlist and the test asserting that stays exact. Forgetting to list a new
#: public route here costs an external key a harmless 403, never access.
EXTERNAL_PUBLIC_OPERATIONS: frozenset[tuple[str, str]] = frozenset({
    ("GET", "/health"),
    ("GET", MUSE_SPEC_PATH),
})

#: Prefix of the server-side origin tag (change 4). ``source:muse`` marks every write made
#: with the muse key, so a Claude Code session on the devvm can see which claims came from
#: an external assistant.
ORIGIN_TAG_PREFIX = "source:"


def is_allowed_for_external(method: str, path: str) -> bool:
    """Whether an "external" key may call this operation."""
    operation = (method.upper(), path)
    return operation in EXTERNAL_ALLOWED_OPERATIONS or operation in EXTERNAL_PUBLIC_OPERATIONS


#: Codepoints that draw like one of the seven characters of ``source:`` but are a
#: different character, so a tag spelled with one reads as provenance to a human while
#: slipping a plain string test. NFKC already folds the fullwidth and mathematical forms
#: (``source：wizard`` among them); what is left is the Cyrillic, Greek and Armenian
#: letters and the three colons that have no compatibility decomposition. Scoped to the
#: characters of our own marker, not a general confusables table.
_ORIGIN_CONFUSABLES = str.maketrans({
    "ѕ": "s",  # CYRILLIC SMALL LETTER DZE
    "о": "o",  # CYRILLIC SMALL LETTER O
    "ο": "o",  # GREEK SMALL LETTER OMICRON
    "օ": "o",  # ARMENIAN SMALL LETTER OH
    "υ": "u",  # GREEK SMALL LETTER UPSILON
    "ս": "u",  # ARMENIAN SMALL LETTER SEH
    "г": "r",  # CYRILLIC SMALL LETTER GHE
    "ր": "r",  # ARMENIAN SMALL LETTER REH
    "с": "c",  # CYRILLIC SMALL LETTER ES
    "ϲ": "c",  # GREEK LUNATE SIGMA SYMBOL
    "е": "e",  # CYRILLIC SMALL LETTER IE
    "∶": ":",  # RATIO
    "꞉": ":",  # MODIFIER LETTER COLON
    "׃": ":",  # HEBREW PUNCTUATION SOF PASUQ
})


def _split_tags(tags: str | None) -> list[str]:
    """The comma-separated tags column as a list: empty elements dropped, each element's
    runs of whitespace collapsed to one space so no tag can break the single line the
    recall hook renders them into."""
    return [tag for tag in (" ".join(raw.split()) for raw in (tags or "").split(",")) if tag]


def origin_tag(user_id: str) -> str:
    """The origin tag for a key belonging to ``user_id`` — the muse key gets source:muse.

    ``user_id`` carries no comma: ``auth._validate_user_id`` drops an API_KEYS entry whose
    id has one, because the tags column is comma-separated and the stamp has to be one
    element of it for ``stamp_origin`` to stay idempotent.
    """
    return f"{ORIGIN_TAG_PREFIX}{user_id}"


def _origin_comparison_form(tag: str) -> str:
    """``tag`` reduced to the form the origin test compares against.

    Compatibility-normalised, case-folded, lookalikes folded to their ASCII twin, then
    every space and control character removed — so ``Source: wizard``, ``source\\twizard``,
    ``source\\xa0:wizard``, ``source：wizard`` and ``ѕource:wizard`` all reduce to text
    containing ``source:``.
    """
    folded = unicodedata.normalize("NFKC", tag).casefold().translate(_ORIGIN_CONFUSABLES)
    return "".join(
        ch for ch in folded if not ch.isspace() and unicodedata.category(ch)[0] != "C"
    )


def _claims_an_origin(tag: str) -> bool:
    """Whether a client-supplied tag could be read as an origin claim, in any spelling.

    Provenance is asserted by the server, so a client-written ``Source: wizard`` must not
    survive to be rendered as provenance by the recall hook.

    Matched on CONTAINS rather than starts-with, because the recall hook renders tags into
    one line (``Tags: a,b``) and a tag carrying a newline splits that line in two: the
    text after the break then reads as a field of its own, so ``note\\ntags: source:wizard``
    paints a forged marker above the genuine one without the claim ever starting the tag.
    Reducing the tag first removes the break, and the contains test catches what is left.

    Near misses are kept: ``sources:muse`` has no ``source:`` in it and stays.
    """
    return ORIGIN_TAG_PREFIX in _origin_comparison_form(tag)


def stamp_origin(tags: str | None, user_id: str) -> str:
    """Return ``tags`` carrying source:<user_id> exactly once, at the end.

    Tags are one comma-separated string (``permissions.py`` reads them with
    ``string_to_array(m.tags, ',')``), so the merge is a string operation:

    * no client tags, or only whitespace, gives the stamp alone;
    * client tags are kept in order, with empty elements dropped and every run of
      whitespace collapsed to one space — the trimming the list and tag-count endpoints
      already apply when reading, extended to the newlines and tabs that would otherwise
      break the single line the recall hook renders tags into;
    * any origin claim the client sent is removed first, so a repeated update cannot
      accumulate the tag twice and a forged ``source:wizard`` cannot survive.

    Idempotent: stamping an already-stamped string returns it unchanged.
    """
    kept = [
        tag
        for tag in _split_tags(tags)
        if not _claims_an_origin(tag)
    ]
    kept.append(origin_tag(user_id))
    return ",".join(kept)


def stamp_stored_origin(tags: str | None, user_id: str) -> str:
    """Add the stamp to a tags value the SERVER read back, leaving the rest untouched.

    Separate from ``stamp_origin`` because the two inputs deserve different treatment. A
    value the client sent may be forging provenance, so every origin claim in it goes. A
    value already in the column was written by somebody else's key — an external key can
    reach a row it does not own through a write-share, and an update that only asked to
    change importance would otherwise delete the provenance on that row.

    Idempotent, and it never writes the stamp twice.
    """
    kept = _split_tags(tags)
    stamp = origin_tag(user_id)
    if stamp not in kept:
        kept.append(stamp)
    return ",".join(kept)
