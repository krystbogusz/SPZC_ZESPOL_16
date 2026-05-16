import re
from urllib.parse import urlparse


class FeatureExtractor:
    def extract_features(self, url):
        parsed = urlparse(url)
        return {
            "url_to_path_length": self.get_url_to_path_length(parsed),
            "has_ip": self.has_ip(parsed),
            "hostname_length": self.get_hostname_length(parsed),
            "has_www": self.has_www(parsed),
            "has_tld": self.has_tld(parsed),
            "has_decimal_in_hostname": self.has_decimal_in_hostname(parsed),
            "path_length": self.get_path_length(parsed),
            "num_subdirectories": self.get_num_subdirectories(parsed),
            "longest_subdirectory_length": self.get_longest_sub_length(parsed),
            "has_date_in_path": self.has_date_in_path(parsed),
            "has_hex_in_path": self.has_hex_in_path(parsed),
            "has_at_symbol": self.has_at_symbol(url),
            "num_dots_in_hostname": self.get_num_dots_in_hostname(parsed),
            "has_hyphen_in_hostname": self.has_hyphen_in_hostname(parsed),
            "has_port": self.has_port(parsed),
        }

    def get_url_to_path_length(self, parsed):
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return len(base_url)

    def has_ip(self, parsed):
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        target = f"{parsed.netloc}{parsed.path}"
        return 1 if re.search(ip_pattern, target) else 0

    def get_hostname_length(self, parsed):
        return len(parsed.netloc)

    def has_www(self, parsed):
        return 1 if parsed.netloc.startswith("www.") else 0

    def has_tld(self, parsed):
        parts = parsed.netloc.split(".")
        return 1 if len(parts) > 1 and parts[-1].isalpha() else 0

    def has_decimal_in_hostname(self, parsed):
        return 1 if re.search(r"\d", parsed.netloc) else 0

    def get_path_length(self, parsed):
        return len(parsed.path)

    def get_num_subdirectories(self, parsed):
        subdirs = [s for s in parsed.path.split("/") if s]
        return len(subdirs)

    def get_longest_sub_length(self, parsed):
        subdirs = [s for s in parsed.path.split("/") if s]
        if not subdirs:
            return 0
        return len(max(subdirs, key=len))

    def has_date_in_path(self, parsed):
        date_pattern = r"(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12]\d|3[01])"
        return 1 if re.search(date_pattern, parsed.path) else 0

    def has_hex_in_path(self, parsed):
        hex_pattern = r"0x[0-9a-fA-F]+"
        return 1 if re.search(hex_pattern, parsed.path) else 0

    def has_at_symbol(self, url):
        return 1 if "@" in url else 0

    def get_num_dots_in_hostname(self, parsed):
        return parsed.netloc.count(".")

    def has_hyphen_in_hostname(self, parsed):
        return 1 if "-" in parsed.netloc else 0

    def has_port(self, parsed):
        return 1 if re.search(r":\d+$", parsed.netloc) else 0
