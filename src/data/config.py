from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _coerce_bbox_to_str(v: object) -> str:
    """Convierte tuple/list a CSV string; pasa strings tal cual.

    pydantic-settings llama json.loads() en campos con tipo complejo (tuple,
    list) antes de pasar el valor a cualquier validador. Al guardar el bbox
    como str se evita ese intento de parseo JSON.
    """
    if isinstance(v, (list, tuple)):
        return ",".join(str(float(x)) for x in v)
    return str(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env.local sobreescribe .env (el último gana)
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Permite pasar tanto el nombre del campo como el alias en el constructor
        populate_by_name=True,
    )

    # --- Credenciales de APIs ---
    firms_map_key: str = Field(default="", description="NASA FIRMS MAP key")
    earthdata_user: str = Field(default="", description="NASA Earthdata username")
    earthdata_pass: str = Field(default="", description="NASA Earthdata password")
    cds_api_key: str = Field(default="", description="Copernicus CDS API key (uid:key)")

    # --- Google Earth Engine ---
    gee_service_account: str = Field(default="", description="GEE service account email")
    gee_private_key_path: Path = Field(
        default=Path(""),
        description="Ruta al JSON de clave privada del service account GEE",
    )

    @property
    def gee_ready(self) -> bool:
        """True si las credenciales GEE están configuradas y el archivo de clave existe."""
        return (
            bool(self.gee_service_account)
            and self.gee_private_key_path != Path("")
            and self.gee_private_key_path.exists()
        )

    # --- Área de estudio ---
    # Guardado como str para evitar que pydantic-settings intente json.loads().
    # Env var: PETEN_BBOX=-91.45,15.88,-89.14,17.82
    # Acceso: cfg.peten_bbox → tuple[float, float, float, float]
    peten_bbox_str: Annotated[str, BeforeValidator(_coerce_bbox_to_str)] = Field(
        default="-91.45,15.88,-89.14,17.82",
        validation_alias="peten_bbox",
        description="Bounding box del Petén en WGS84: lon_min,lat_min,lon_max,lat_max",
    )
    grid_resolution_km: float = Field(
        default=1.0,
        gt=0,
        description="Resolución de la grilla en kilómetros",
    )

    # --- CRS ---
    crs_work: str = Field(
        default="EPSG:32616",
        description="CRS de trabajo para análisis métrico (UTM zona 16N)",
    )
    crs_geo: str = Field(
        default="EPSG:4326",
        description="CRS geográfico para I/O y mapas web",
    )

    # --- Rutas ---
    project_root: Path = Field(default=Path("."), description="Raíz del proyecto")

    @property
    def data_raw(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def data_interim(self) -> Path:
        return self.project_root / "data" / "interim"

    @property
    def data_processed(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def peten_bbox(self) -> tuple[float, float, float, float]:
        """Retorna el bounding box del Petén como tupla (lon_min, lat_min, lon_max, lat_max)."""
        parts = [float(p.strip()) for p in self.peten_bbox_str.split(",")]
        return (parts[0], parts[1], parts[2], parts[3])

    # --- Pipeline ---
    random_seed: int = Field(default=42, description="Semilla global para reproducibilidad")
    log_level: str = Field(default="INFO", description="Nivel de logging de loguru")

    # --- Período de datos ---
    train_start: str = Field(default="2018-01-01")
    train_end: str = Field(default="2022-12-31")
    val_start: str = Field(default="2023-01-01")
    val_end: str = Field(default="2023-12-31")
    test_start: str = Field(default="2024-01-01")
    test_end: str = Field(default="2024-12-31")

    @model_validator(mode="after")
    def _validate_bbox(self) -> Settings:
        parts = [p.strip() for p in self.peten_bbox_str.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"PETEN_BBOX debe tener exactamente 4 valores separados por coma, "
                f"recibido: {self.peten_bbox_str!r}"
            )
        try:
            [float(p) for p in parts]
        except ValueError as exc:
            raise ValueError(
                f"PETEN_BBOX contiene valores no numéricos: {self.peten_bbox_str!r}"
            ) from exc
        return self


# Instancia global — usar en todos los módulos
settings = Settings()
