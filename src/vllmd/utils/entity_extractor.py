"""
Entity extraction from text using user-defined rules
(regex patterns and keyword lists). Use for frame descriptions or any input string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class Entity:
    """A single named entity (text, label, span)."""

    text: str
    label: str
    start: int
    end: int

    def __str__(self) -> str:
        return f"{self.text} ({self.label})"


@dataclass
class CustomEntityRule:
    """Rule for one custom entity type: either a regex pattern or a list of keywords."""

    label: str
    pattern: Optional[str] = None   # regex, one capturing group = entity span
    keywords: Optional[List[str]] = None  # exact (or case-insensitive) phrases
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.pattern is None and not self.keywords:
            raise ValueError("Provide either pattern or keywords for CustomEntityRule")


class CustomEntityExtractor:
    """
    Extracts custom entities from text using user-defined rules:
    - Regex patterns (one group = entity text)
    - Keyword lists (exact or case-insensitive phrase match)

    Use for domain-specific labels (e.g. ACTION, OBJECT, SCENE) on frame descriptions.
    """

    def __init__(
        self,
        rules: Optional[List[CustomEntityRule]] = None,
        merge_overlaps: bool = True,
    ) -> None:
        """
        Args:
            rules: List of CustomEntityRule. If None, use add_rule() later.
            merge_overlaps: If True, drop shorter overlapping spans (keep first by start).
        """
        self._rules: List[CustomEntityRule] = rules or []
        self._compiled: Dict[str, re.Pattern] = {}
        self.merge_overlaps = merge_overlaps
        for r in self._rules:
            if r.pattern:
                self._compiled[r.label] = re.compile(r.pattern, 0 if r.case_sensitive else re.IGNORECASE)

    def add_rule(self, rule: CustomEntityRule) -> None:
        """Add a rule (regex compiled on first use)."""
        self._rules.append(rule)
        if rule.pattern:
            self._compiled[rule.label] = re.compile(
                rule.pattern, 0 if rule.case_sensitive else re.IGNORECASE
            )

    def add_keyword_rule(self, label: str, keywords: List[str], case_sensitive: bool = False) -> None:
        """Convenience: add a rule that matches any of the given keywords."""
        self.add_rule(
            CustomEntityRule(label=label, keywords=keywords, case_sensitive=case_sensitive)
        )

    def add_regex_rule(self, label: str, pattern: str, case_sensitive: bool = False) -> None:
        """Convenience: add a rule with a regex. Use one capturing group for the entity span."""
        self.add_rule(
            CustomEntityRule(label=label, pattern=pattern, case_sensitive=case_sensitive)
        )

    def extract(self, text: str) -> List[Entity]:
        """
        Extract custom entities from text.

        Returns:
            List of Entity(text, label, start, end), optionally with overlaps merged.
        """
        if text is None or not text.strip():
            return []
        entities: List[Entity] = []
        for rule in self._rules:
            if rule.pattern:
                pat = self._compiled.get(rule.label)
                if pat:
                    for m in pat.finditer(text):
                        group = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                        entities.append(
                            Entity(text=group, label=rule.label, start=m.start(), end=m.end())
                        )
            if rule.keywords:
                lower_text = text if rule.case_sensitive else text.lower()
                for kw in rule.keywords:
                    search = kw if rule.case_sensitive else kw.lower()
                    start = 0
                    while True:
                        idx = lower_text.find(search, start)
                        if idx < 0:
                            break
                        entities.append(
                            Entity(text=text[idx : idx + len(kw)], label=rule.label, start=idx, end=idx + len(kw))
                        )
                        start = idx + 1
        if self.merge_overlaps and entities:
            entities = self._merge_overlapping(entities)
        return entities

    def _merge_overlapping(self, entities: List[Entity]) -> List[Entity]:
        """Keep non-overlapping spans; on overlap keep the one that starts first (then longer)."""
        if not entities:
            return []
        sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
        out: List[Entity] = []
        for e in sorted_ents:
            if not out or e.start >= out[-1].end:
                out.append(e)
        return out

    def extract_batch(
        self,
        texts: List[str],
        merge_duplicates: bool = False,
    ) -> List[List[Entity]]:
        """Extract custom entities from multiple texts."""
        result: List[List[Entity]] = []
        for text in texts:
            entities = self.extract(text)
            if merge_duplicates:
                seen = set()
                unique: List[Entity] = []
                for e in entities:
                    key = (e.text.strip(), e.label)
                    if key not in seen:
                        seen.add(key)
                        unique.append(e)
                result.append(unique)
            else:
                result.append(entities)
        return result


def load_rules_from_file(
    path: Union[str, Path],
    merge_overlaps: bool = True,
) -> CustomEntityExtractor:
    """
    Load entity rules from a structured text file and return a configured
    CustomEntityExtractor.

    File format (entity_rules.txt):
      [LABEL]
      type = keywords | regex
      keywords = item1, item2, ...   (when type=keywords)
      pattern = \\b(...)\\b           (when type=regex)
      case_sensitive = true | false  (optional, default false)

    Lines starting with # are ignored. Blank lines and section headers [LABEL]
    define blocks.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Rules file not found: {path}")

    extractor = CustomEntityExtractor(merge_overlaps=merge_overlaps)
    content = path.read_text(encoding="utf-8")
    current_label: Optional[str] = None
    current_type: Optional[str] = None
    current_keywords: List[str] = []
    current_pattern: Optional[str] = None
    case_sensitive = False

    def flush_rule() -> None:
        nonlocal current_label, current_type, current_keywords, current_pattern, case_sensitive
        if not current_label:
            return
        if current_type == "keywords" and current_keywords:
            extractor.add_keyword_rule(
                current_label, current_keywords, case_sensitive=case_sensitive
            )
        elif current_type == "regex" and current_pattern:
            extractor.add_regex_rule(
                current_label, current_pattern, case_sensitive=case_sensitive
            )
        current_label = None
        current_type = None
        current_keywords = []
        current_pattern = None
        case_sensitive = False

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            flush_rule()
            current_label = line[1:-1].strip()
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip().lower(), value.strip()
            if key == "type":
                current_type = value.lower()
            elif key == "keywords":
                current_keywords = [k.strip() for k in value.split(",") if k.strip()]
            elif key == "pattern":
                current_pattern = value
            elif key == "case_sensitive":
                case_sensitive = value.lower() in ("true", "1", "yes")

    flush_rule()
    return extractor
