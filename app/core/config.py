from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Anomalib (Engine)
    anomalib_ckpt_path: str = Field(default="./models/model.ckpt", alias="ANOMALIB_CKPT_PATH")
    anomalib_model_class: str = Field(default="Patchcore", alias="ANOMALIB_MODEL_CLASS")
    anomalib_device: str = Field(default="auto", alias="ANOMALIB_DEVICE")  # auto/cpu/cuda

    # OpenAI
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5", alias="OPENAI_MODEL")
    openai_instructions: str = Field(default="You are a helpful assistant.", alias="OPENAI_INSTRUCTIONS")

settings = Settings()
