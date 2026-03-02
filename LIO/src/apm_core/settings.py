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

    # MSSQL tables (switch with SQLALCHEMY / SQLALCHEMY_DEV)
    mssql_table: str               # APM_Scalability or APM_Scalability_Dev
    mssql_events_table: str        # NewApmModelEvents or NewApmModelEvents_Dev

    # Minocore API (raw pull) – kept for compatibility even if not used
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
      - write scores to SQLALCHEMY_DEV.table
      - write events to SQLALCHEMY_DEV.events_table
      - use Email_DEV section for sending
    debug=False means PROD equivalents.
    """
    ini_path = site_root / "etc" / "config.ini"
    cfg = ConfigParser()
    cfg.read(ini_path)

    # --- Postgres ---
    pg_host = cfg.get("DATABASE", "host", fallback="")
    pg_user = cfg.get("DATABASE", "user", fallback="")
    pg_dbname = cfg.get("DATABASE", "dbname", fallback="")
    pg_password = cfg.get("DATABASE", "password", fallback="")
    pg_port = int(cfg.get("DATABASE", "port", fallback="5432"))

    # --- MSSQL section selection ---
    sql_section = "SQLALCHEMY_DEV" if debug else "SQLALCHEMY"

    mssql_host = cfg.get(sql_section, "host")
    mssql_dbname = cfg.get(sql_section, "dbname")
    mssql_user = cfg.get(sql_section, "user")
    mssql_password = cfg.get(sql_section, "password")
    mssql_port = int(cfg.get(sql_section, "port", fallback="1433"))

    # Score table (APM_Scalability / APM_Scalability_Dev)
    mssql_table = cfg.get(sql_section, "table")

    # Events table (NewApmModelEvents / NewApmModelEvents_Dev)
    # fallback keeps backward compatibility if not yet added to config.ini
    mssql_events_table = cfg.get(sql_section, "events_table", fallback="NewApmModelEvents")

    # --- Minocore API ---
    minocore_api_url = cfg.get("Minocore_API", "api_url", fallback="")
    minocore_bearer_token = cfg.get("Minocore_API", "bearer_token", fallback="")
    minocore_entity_id = cfg.get("Minocore_API", "entity_id", fallback="")

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
        mssql_events_table=mssql_events_table,
        minocore_api_url=minocore_api_url,
        minocore_bearer_token=minocore_bearer_token,
        minocore_entity_id=minocore_entity_id,
        cfg=cfg,
    )