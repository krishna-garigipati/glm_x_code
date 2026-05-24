LANGUAGE_CODE_COMMON_NAMES = {
    "en": "english", "fr": "french", "de": "german", "es": "spanish",
    "it": "italian", "pt": "portuguese", "nl": "dutch", "ru": "russian",
    "zh": "chinese", "ja": "japanese", "ko": "korean", "ar": "arabic",
}


def detect_language_from_uri(uri: str) -> str:
    parts = uri.strip("/").split("/")
    if len(parts) >= 5 and parts[0] == "http:":
        return parts[4]
    return "unknown"


def is_english_uri(uri: str) -> bool:
    return detect_language_from_uri(uri) == "en"


def strip_conceptnet_uri(uri: str) -> str:
    parts = uri.strip("/").split("/")
    name_idx = 4 if parts[0] == "http:" else 3
    if len(parts) > name_idx:
        return parts[name_idx].replace("_", " ")
    return uri
