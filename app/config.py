# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Loads and validates application settings from environment variables."""
    data_file_path: str
    members_file_path: str
    shops_file_path: str
    cities_file_path: str
    categories_file_path: str
    searches_file_path: str

    scenario7_keylist_path: str | None = "label.txt"

    openai_api_key: str
    openai_base_url: str
    openai_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file="./.env", env_file_encoding="utf-8")

settings = Settings()