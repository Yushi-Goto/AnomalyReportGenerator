from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anomalib
    anomalib_model_path: str = Field(default="./models/model.pt", alias="ANOMALIB_MODEL_PATH")
    anomalib_metadata_path: str = Field(default="./models/metadata.json", alias="ANOMALIB_METADATA_PATH")
    anomalib_device: str = Field(default="auto", alias="ANOMALIB_DEVICE")  # auto/cpu/cuda

    # OpenAI
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5", alias="OPENAI_MODEL")
    openai_instructions: str = Field(default="You are a helpful assistant.", alias="OPENAI_INSTRUCTIONS")

settings = Settings()
