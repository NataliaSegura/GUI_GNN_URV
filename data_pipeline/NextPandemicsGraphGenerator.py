import os

import torch

from pathlib import Path
from torch_geometric.data import Data
from tqdm import tqdm
from data_pipeline.pic50_utils import load_pic50

from job_config.nextpandemics.NextPandemicsDataConfig import NextPandemicsDataConfig
from data_pipeline.pic50_utils import load_pic50
from data_pipeline.nextpandemics_ligand import LigandCollection

class NextPandemicsGraphGenerator:

    def __init__(self, config):
        self.config = config

    def _build_ligand_graphs(self, collection, pdb_id, pic50, min_coord, max_coord):
        """Single-sample graph creation."""
        #print(f"Building ligand graph for {pdb_id} with pIC50 {pic50}")
        sdf_path = os.path.join(str(self.config.ligand_path),"Ligand_SDF",f"{pdb_id}_ligand.sdf")
        collection.add_ligand(sdf_path, pdb_id, min_coord, max_coord)
       # ligand = collection.ligands.get(pdb_id)
        if collection.ligands.get(pdb_id) is None:
            return None

        # convert ligand into Data(...)

    def _use_pocket_mode(self):

        return (
            self.config.protein_path is not None
            and self.config.interaction_path is not None
            and self.config.distance_threshold is not None
        )

    def build_graphs_pt(self):
        min_coord = [-28, -36, -34]
        max_coord = [39, 37, 42]

        pic50_dict = load_pic50(self.config.pic50_path)
        if self.config.output_path is None:
            self.output_path = Path("outputs/nextpandemics/data")
        else: 
            self.output_path = Path(self.config.output_path)

        self.output_path.mkdir(parents=True, exist_ok=True)

        if self._use_pocket_mode():
            self._build_pocket_graphs()

        else:
            collection = LigandCollection()
            for pdb_id, pic50 in tqdm(pic50_dict.items(), desc="Building ligand graphs"):
                data = self._build_ligand_graphs(collection, pdb_id, pic50, min_coord, max_coord)


