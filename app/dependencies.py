# app/dependencies.py
import pandas as pd
from functools import lru_cache
from .config import settings

# ---------------- Products ----------------
@lru_cache(maxsize=1)
def get_products_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.data_file_path)
        cols = [c for c in ["persian_name", "random_key", "extra_features", "category_id"] if c in df.columns]
        if cols:
            df = df[cols].copy()
        # normalize dtypes
        if "random_key" in df.columns:
            df["random_key"] = df["random_key"].astype(str)
        if "persian_name" in df.columns:
            df["persian_name"] = df["persian_name"].astype(str)
        if "category_id" in df.columns:
            df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").astype("Int64")
        return df
    except FileNotFoundError:
        print(f"WARNING: Data file not found at '{settings.data_file_path}'.")
        return pd.DataFrame(columns=["persian_name", "random_key", "category_id"])

# ---------------- Members ----------------
@lru_cache(maxsize=1)
def get_members_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.members_file_path)

        # price → numeric
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # normalize key columns if present
        if "base_random_key" in df.columns:
            df["base_random_key"] = df["base_random_key"].astype(str)

        if "shop_id" in df.columns:
            df["shop_id"] = pd.to_numeric(df["shop_id"], errors="coerce").astype("Int64")

        if "random_key" in df.columns:
            df["random_key"] = df["random_key"].astype(str)

        return df
    except FileNotFoundError:
        print(f"WARNING: Members data file not found at '{settings.members_file_path}'.")
        return pd.DataFrame(columns=["base_random_key", "price", "shop_id", "random_key"])

# ---------------- Shops ----------------
@lru_cache(maxsize=1)
def get_shops_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.shops_file_path)

        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        if "city_id" in df.columns:
            df["city_id"] = pd.to_numeric(df["city_id"], errors="coerce").astype("Int64")
        if "has_warranty" in df.columns:
            df["has_warranty"] = df["has_warranty"].astype(bool)
        else:
            df["has_warranty"] = False

        return df
    except FileNotFoundError:
        print(f"WARNING: Shops data file not found at '{settings.shops_file_path}'.")
        return pd.DataFrame(columns=["id", "city_id", "has_warranty"])

# ---------------- Cities ----------------
@lru_cache(maxsize=1)
def get_cities_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.cities_file_path)
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        if "name" in df.columns:
            df["name"] = df["name"].astype(str)
        return df
    except FileNotFoundError:
        print(f"WARNING: Cities data file not found at '{settings.cities_file_path}'.")
        return pd.DataFrame(columns=["id", "name"])

# ---------------- Categories ----------------
@lru_cache(maxsize=1)
def get_categories_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.categories_file_path)
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        if "title" in df.columns:
            df["title"] = df["title"].astype(str)
        return df
    except FileNotFoundError:
        print(f"WARNING: Categories data file not found at '{settings.cities_file_path}'.")
        return pd.DataFrame(columns=["id", "title"])

# ---------------- Searches ----------------
@lru_cache(maxsize=1)
def get_searches_df() -> pd.DataFrame:

    try:
        df = pd.read_parquet(settings.searches_file_path)
        if "result_base_product_rks" in df.columns:
            def _to_list(v):
                if v is None: return []
                if isinstance(v, list): return [str(x) for x in v]
                try:
                    import json
                    x = json.loads(str(v))
                    return [str(i) for i in (x if isinstance(x, list) else [])]
                except Exception:
                    return []
            df["result_base_product_rks"] = df["result_base_product_rks"].map(_to_list)
        if "query" in df.columns:
            df["query"] = df["query"].astype(str)
        if "category_id" in df.columns:
            df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").astype("Int64")
        return df
    except FileNotFoundError:
        print("WARNING: searches parquet not found.")
        return pd.DataFrame(columns=["query","result_base_product_rks","category_id"])