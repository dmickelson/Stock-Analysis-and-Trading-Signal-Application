from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    db_path: str
    log_level: str

    class Config:
        env_file = ".env"


settings = Settings()
