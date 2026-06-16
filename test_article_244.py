#!/usr/bin/env python
"""Test article retrieval for article 244."""
from config import settings
from core.db_connector import get_article_by_id

# Test article 244
article = get_article_by_id(244)
if article:
    print("=== ARTICLE 244 ===")
    print(f"Title: {article.get('title', 'NO TITLE')}")
    content = article.get('content', '')
    print(f"Content length: {len(content)}")
    print(f"\nFirst 500 chars:\n{content[:500]}")
    
    # Count lines
    lines = content.split('\n')
    print(f"\nTotal lines: {len(lines)}")
    print(f"\nFirst 10 non-empty lines:")
    count = 0
    for i, line in enumerate(lines, 1):
        if line.strip():
            print(f"{i}: {line[:100]}")
            count += 1
            if count >= 10:
                break
else:
    print("ERROR: Article 244 not found!")
