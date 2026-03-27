import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from backend.api.schemas import UploadResponse
from backend.core.config import settings


class UploadStore:
    def __init__(self) -> None:
        self.root = settings.uploads_dir

    def save_upload(self, filename: str, content: bytes) -> UploadResponse:
        upload_id = uuid4().hex[:12]
        safe_name = Path(filename).name or "structure.cif"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in {".cif", ".vasp", ".poscar", ".txt"}:
            suffix = ".cif"

        structure_type = "POSCAR" if suffix in {".vasp", ".poscar"} else "CIF"
        path = self.root / f"{upload_id}{suffix}"
        path.write_bytes(content)

        text = content.decode("utf-8", errors="ignore")
        payload = {
            "upload_id": upload_id,
            "filename": safe_name,
            "structure_type": structure_type,
            "formula_hint": self._extract_formula_hint(text),
            "preview": self._preview_text(text),
            "path": str(path),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (self.root / f"{upload_id}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return UploadResponse(
            upload_id=payload["upload_id"],
            filename=payload["filename"],
            structure_type=payload["structure_type"],
            formula_hint=payload["formula_hint"],
            preview=payload["preview"],
        )

    def get_upload(self, upload_id: str) -> dict:
        meta_path = self.root / f"{upload_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Upload '{upload_id}' was not found.")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    @staticmethod
    def _preview_text(text: str, max_lines: int = 6) -> str:
        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[:max_lines])

    @staticmethod
    def _extract_formula_hint(text: str) -> Optional[str]:
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("data_") and len(line) > 5:
                return line[5:].strip() or None
            if "chemical_formula_sum" in line.lower():
                parts = line.split()
                if parts:
                    return parts[-1].strip("'\"")
        return None
