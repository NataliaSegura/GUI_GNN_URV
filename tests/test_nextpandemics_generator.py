from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_pipeline.NextPandemicsGraphGenerator import NextPandemicsGraphGenerator
from job_config.nextpandemics.NextPandemicsDataConfig import  NextPandemicsDataConfig

config = NextPandemicsDataConfig(
    pic50_path=Path("C:\\Users\\natal\\code_binding_data_and_results\\data\\Mpro-URV\\pIC50.txt"),
    ligand_path=Path("C:\\Users\\natal\\code_binding_data_and_results\\data\\Mpro-URV\\Ligand"),
    protein_path=None,
    interaction_path=None,
    output_path=Path("output_test_nextpandemics")
)

generator = NextPandemicsGraphGenerator(config)

generator.build_graphs_pt()