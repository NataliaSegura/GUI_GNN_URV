from dataclasses import dataclass
from pathlib import Path


@dataclass
class NextPandemicsDataConfig:
    pic50_path: Path
    ligand_path: Path
    protein_path: Path
    interaction_path: Path
    output_path: Path
    dis_threshold: float = 5.0
    
