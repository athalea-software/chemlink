import math

import pytest

from chemlink.utils.receptor_processor import ReceptorProcessor


TNKS1_TERMINUS = """\
ATOM      1  N   PRO A1313      15.219 -40.891 -27.137  1.00  0.00           N  
ATOM      2  CA  PRO A1313      13.987 -41.514 -27.635  1.00  0.00           C  
ATOM      3  C   PRO A1313      14.201 -42.960 -28.075  1.00  0.00           C  
ATOM      4  O   PRO A1313      13.277 -43.765 -27.955  1.00  0.00           O  
ATOM      5  CB  PRO A1313      13.605 -40.633 -28.830  1.00  0.00           C  
ATOM      6  CG  PRO A1313      14.271 -39.325 -28.566  1.00  0.00           C  
ATOM      7  CD  PRO A1313      15.559 -39.670 -27.887  1.00  0.00           C  
ATOM      8  OXT PRO A1313      13.099 -42.572 -27.585  1.00  0.00           O  
TER
END
"""


def _coordinates(line):
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def test_repairs_invalid_terminal_oxt_from_tnks1(tmp_path):
    receptor = tmp_path / "tnks1.pdb"
    receptor.write_text(TNKS1_TERMINUS)

    repaired = ReceptorProcessor._repair_invalid_terminal_oxt(str(receptor))

    assert repaired == [("A", "1313")]
    atoms = {
        line[12:16].strip(): _coordinates(line)
        for line in receptor.read_text().splitlines()
        if line.startswith("ATOM")
    }
    assert math.dist(atoms["C"], atoms["OXT"]) == pytest.approx(1.25, abs=0.01)
    assert math.dist(atoms["O"], atoms["OXT"]) > 1.70
    assert math.dist(atoms["CA"], atoms["OXT"]) > 1.70


def test_records_terminal_oxt_repair_in_preparation_log(tmp_path):
    receptor = tmp_path / "tnks1_h.pdb"
    warning_log = tmp_path / "receptor_preparation.log"
    receptor.write_text(TNKS1_TERMINUS)

    ReceptorProcessor._repair_invalid_terminal_oxt(
        str(receptor), warning_log=str(warning_log)
    )

    log_entry = warning_log.read_text()
    assert "WARNING | receptor_preparation | tnks1_h.pdb" in log_entry
    assert "Rebuilt invalid terminal OXT at A:1313" in log_entry
    assert "old C-OXT" in log_entry
    assert "old O-OXT" in log_entry


def test_preserves_valid_terminal_oxt(tmp_path):
    receptor = tmp_path / "valid.pdb"
    valid = TNKS1_TERMINUS.replace(
        "13.099 -42.572 -27.585", "15.307 -43.324 -28.553"
    )
    receptor.write_text(valid)

    before = receptor.read_text()
    repaired = ReceptorProcessor._repair_invalid_terminal_oxt(str(receptor))

    assert repaired == []
    assert receptor.read_text() == before
