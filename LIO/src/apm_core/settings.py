# APM/LIO/src/apm_core/settings.py
from __future__ import annotations

from dataclasses import dataclass
from configparser import ConfigParser
from pathlib import Path


@dataclass
class IniSettings:
    # Postgres (legacy / optional)
    pg_host: str
    pg_user: str
    pg_dbname: str
    pg_password: str
    pg_port: int

    # MSSQL
    mssql_host: str
    mssql_dbname: str
    mssql_user: str
    mssql_password: str
    mssql_port: int
    mssql_table: str

    # Minocore API (raw pull)
    minocore_api_url: str
    minocore_bearer_token: str
    minocore_entity_id: str

    # Keep original parsed config so other modules can read sections safely
    cfg: ConfigParser

    def get(self, section: str, option: str, fallback=None):
        """Compatibility helper so code can call settings.get(section, option)."""
        if not self.cfg.has_section(section):
            return fallback
        return self.cfg.get(section, option, fallback=fallback)


def load_ini(site_root: Path, debug: bool) -> IniSettings:
    """
    debug=True means:
      - write to SQLALCHEMY_DEV table
      - use Email_DEV section for sending
    debug=False means PROD equivalents.
    """
    ini_path = site_root / "etc" / "config.ini"
    cfg = ConfigParser()
    cfg.read(ini_path)

    # --- Postgres (may not be used if you pull from Minocore) ---
    pg_host = cfg.get("DATABASE", "host", fallback="")
    pg_user = cfg.get("DATABASE", "user", fallback="")
    pg_dbname = cfg.get("DATABASE", "dbname", fallback="")
    pg_password = cfg.get("DATABASE", "password", fallback="")
    pg_port = int(cfg.get("DATABASE", "port", fallback="5432"))

    # --- MSSQL ---
    sql_section = "SQLALCHEMY_DEV" if debug else "SQLALCHEMY"
    mssql_host = cfg.get(sql_section, "host")
    mssql_dbname = cfg.get(sql_section, "dbname")
    mssql_user = cfg.get(sql_section, "user")
    mssql_password = cfg.get(sql_section, "password")
    mssql_port = int(cfg.get(sql_section, "port", fallback="1433"))
    mssql_table = cfg.get(sql_section, "table")

    # --- Minocore API ---
    minocore_api_url = cfg.get("Minocore_API", "api_url")
    minocore_bearer_token = cfg.get("Minocore_API", "bearer_token")
    minocore_entity_id = cfg.get("Minocore_API", "entity_id")

    return IniSettings(
        pg_host=pg_host,
        pg_user=pg_user,
        pg_dbname=pg_dbname,
        pg_password=pg_password,
        pg_port=pg_port,
        mssql_host=mssql_host,
        mssql_dbname=mssql_dbname,
        mssql_user=mssql_user,
        mssql_password=mssql_password,
        mssql_port=mssql_port,
        mssql_table=mssql_table,
        minocore_api_url=minocore_api_url,
        minocore_bearer_token=minocore_bearer_token,
        minocore_entity_id=minocore_entity_id,
        cfg=cfg,
    )