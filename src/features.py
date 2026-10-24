from urllib.parse import urlparse
import re
import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "confirm",
    "update", "account", "billing", "password",
    "signin", "pay", "invoice", "alert",
    "service", "security", "unlock", "auth"
]

COMMON_LEGIT_TLDS = {
    "com", "org", "net", "edu", "gov",
    "io", "in", "co", "info"
}

BRAND_NAMES = [
    "amazon", "apple", "paypal", "microsoft",
    "google", "facebook", "instagram", "netflix"
]


def normalize_url(url: str) -> str:
    if not isinstance(url, str):
        return ""

    url = url.strip()

    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    return url


def count_suspicious_keywords(url: str) -> int:
    url = url.lower()

    return sum(
        1 for word in SUSPICIOUS_KEYWORDS
        if word in url
    )


def extract_features(url: str) -> dict:
    url = normalize_url(url)
    parsed = urlparse(url)

    domain = parsed.netloc.lower().split("@")[-1]
    domain = domain.split(":")[0]
    path = parsed.path or ""
    query = parsed.query or ""

    length = len(url)
    domain_length = len(domain)

    subdomain_count = max(
        0, len(domain.split(".")) - 2
    )

    suspicious_count = count_suspicious_keywords(url)

    special_characters = sum(
        1 for char in url if char in "@-_=?%&"
    )

    has_ip = bool(
        re.fullmatch(
            r"\d{1,3}(?:\.\d{1,3}){3}",
            domain
        )
    )

    brand_like = any(
        brand in domain for brand in BRAND_NAMES
    )

    brand_in_path = any(
        brand in path.lower() for brand in BRAND_NAMES
    )

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""

    return {
        "url_length": length,
        "domain_length": domain_length,
        "subdomain_count": subdomain_count,
        "suspicious_keyword_count": suspicious_count,
        "special_character_count": special_characters,
        "query_length": len(query),
        "path_length": len(path),
        "has_ip": int(has_ip),
        "https": int(parsed.scheme == "https"),
        "brand_like": int(brand_like),
        "brand_in_path": int(brand_in_path),
        "suspicious_tld": int(
            bool(tld) and tld not in COMMON_LEGIT_TLDS
        ),
    }


def build_feature_dataframe(
    df: pd.DataFrame,
    url_col: str = "url",
    label_col: str = "label"
) -> pd.DataFrame:

    if url_col not in df.columns:
        raise ValueError(f"Missing URL column: {url_col}")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    features = df[url_col].apply(extract_features)

    feature_df = pd.DataFrame(
        features.tolist(),
        index=df.index
    )

    feature_df[label_col] = df[label_col]

    return feature_df


if __name__ == "__main__":
    test_url = "https://example.com/login"
    print(extract_features(test_url))