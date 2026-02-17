


import re
from adarag.run_baselines import Passage
from matplotlib import collections


def _mine_candidates(self, passages: list[Passage], max_candidates: int = 50) -> list[tuple[str, float]]:
        """
        Cheap candidate miner for Stage-1:
          - capitalized phrases (1-4 words)
          - numbers (incl. years)
          - basic date patterns
        Returns list of (candidate_string, prior_score) sorted descending.
        """
        # Weight earlier passages higher (rank prior)
        # Also weight by "closeness": if score is distance, smaller is better.
        # We don’t know your index metric, so we use rank-only by default.
        cand_scores: dict[str, float] = collections.defaultdict(float)

        cap_phrase = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")
        # numbers / years
        num_pat = re.compile(r"\b\d{1,4}\b")
        year_pat = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
        # simple month date mentions
        month_pat = re.compile(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
            re.IGNORECASE,
        )

        for rank, p in enumerate(passages):
            rank_w = 1.0 / (1.0 + rank)

            text = (p.title + "\n" + (p.text or ""))[:20000]  # avoid pathological long docs

            # Capitalized phrases (good for names/orgs/places)
            for m in cap_phrase.findall(text):
                c = m.strip()
                if len(c) < 3:
                    continue
                # filter common sentence starters that pollute candidates
                if c in {"The", "A", "An"}:
                    continue
                cand_scores[c] += 1.0 * rank_w

            # Numbers (good for years, counts)
            for m in num_pat.findall(text):
                cand_scores[m] += 0.5 * rank_w

            for m in year_pat.findall(text):
                cand_scores[m] += 1.0 * rank_w

            # If month appears, try to grab a nearby day/year pattern (very rough)
            if month_pat.search(text):
                # e.g., "January 12, 1999" / "Jan 12 1999"
                date_pat = re.compile(
                    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                    r"\s+\d{1,2}(?:,?\s+\d{4})?\b",
                    re.IGNORECASE,
                )
                for m in date_pat.findall(text):
                    cand_scores[m.strip()] += 1.0 * rank_w

        # Dedup with light normalization: collapse whitespace
        normalized_map: dict[str, str] = {}
        merged: dict[str, float] = collections.defaultdict(float)
        for c, s in cand_scores.items():
            key = " ".join(c.split())
            normalized_map[key] = c  # keep first surface form
            merged[key] += s

        items = sorted(((normalized_map[k], v) for k, v in merged.items()), key=lambda x: x[1], reverse=True)
        return items[:max_candidates]
