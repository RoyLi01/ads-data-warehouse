from __future__ import annotations

"""
Entry script for generating mock ODS data.

This file exists to provide a stable CLI name used by run_all.sh:
- python data_generator/generate_data.py

Internally it delegates to `data_generator/generate_ads_ods.py`.
"""

from data_generator.generate_ads_ods import main


if __name__ == "__main__":
    main()

