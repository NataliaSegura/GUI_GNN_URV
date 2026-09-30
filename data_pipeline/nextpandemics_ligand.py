# data_pipeline/nextpandemics_ligand.py

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from typing import Dict, List, Optional, Tuple
import numpy as np
from data_pipeline.atom_utils import HBD_PATTERN, HBA_PATTERN, FLEXIBILITY_BOND_PATTERN, PERIODIC_ELEMENTS, HYBRIDIZATION, AA_3TO1, BOND_TYPES_COVALENT, BOND_TYPES_NON_COVALENT
import torch


class Ligand:
    """
    Represents a ligand molecule with its graph representation.

    Attributes:
        sdf_path: Path to the SDF file
        mol: RDKit molecule object
        name: Ligand name from PDB
        features: Node features for each atom
        edges: Edge list (connectivity)
        edge_features: Edge features
        coordinates: 3D coordinates (normalized)
        atom_names_pdb: Atom names from PDB file (if available)
        num_atoms: Number of non-hydrogen atoms
    """

    def __init__(self, ligand_path: str, min_coord: list, max_coord: list):
        """
        Initialize a Ligand object from an SDF file.

        Args:
            sdf_path: Path to the SDF file
        """
        self.ligand_path = ligand_path
        self.mol = None
        self.name = None
        self.features = []
        self.edges = []
        self.edge_features = []
        self.coordinates = []
        self.atom_names_pdb = []
        self.num_atoms = 0
        self.index_H_atoms = []

        # Embedding-specific attributes
        self.atom_indices = None
        self.aa_indices = None
        self.hyb_indices = None
        self.bond_indices = None
        self.edge_distances = None
        self.hbd_indices = {}
        self.hba_indices = {}
        self.flexible_edges = set()

        # Load molecule
        self._load_molecule()
        self._build_graph(min_coord, max_coord) 


    def _extract_atom_names_from_cif(self):
        """
        Read a CIF file and extract atom names from the _atom_site.label_atom_id column.
        Returns a list of atom names (e.g., ['N', 'C', 'C1', 'N1', ...]).
        """
        cif_path = self.ligand_path.replace("Ligand_SDF", "Ligand_CIF").replace(".sdf", ".cif")
        if self.ligand_path.split('\\')[-1].split('_')[0] == '9G0I':
            d = 0
        with open(cif_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("HETATM", "ATOM")):
                    fields = line.split()
                    if len(fields) >= 4:  # ensure column exists
                       if not fields[3].startswith('H') and (fields[4] == '.' or fields[4] == 'A'):
                            self.atom_names_pdb.append(fields[3])
                            if self.name == '' or self.name is None or self.name == 'UNK':
                                self.name = fields[5]


    def _load_molecule(self):
        """Load molecule from SDF file."""

        mol_supplier = Chem.SDMolSupplier(self.ligand_path, sanitize=True, removeHs=False)

        self.mol = mol_supplier[0]

        if self.mol is None:
            raise ValueError(f"Invalid SDF file or molecule at: {self.ligand_path}")

        # Extract name
        self.name = self.mol.GetProp("_Name") if bool(self.mol.HasProp("_Name")) else "UNK"
        self._extract_atom_names_from_cif()
        # Compute charges
        rdPartialCharges.ComputeGasteigerCharges(self.mol)

        hbd_matches = self.mol.GetSubstructMatches(HBD_PATTERN)
        self.hbd_indices = {idx[0] for idx in hbd_matches}  # Usamos set para búsqueda rápida O(1)

        hba_matches = self.mol.GetSubstructMatches(HBA_PATTERN)
        self.hba_indices = {idx[0] for idx in hba_matches}
        flexible_matches = self.mol.GetSubstructMatches(FLEXIBILITY_BOND_PATTERN)

        for match in flexible_matches:
            # Añadimos ambas direcciones porque nuestro grafo será bidireccional
            self.flexible_edges.add((match[0], match[1]))
            self.flexible_edges.add((match[1], match[0]))


    def _one_of_k_encoding_unk(self, x, allowable_set: List) -> List[bool]:
        """One-hot encoding with unknown handling."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]


    def _atom_features_one_hot(self, atom) -> np.ndarray:
        """ Extract features for a single atom.
        Args: atom: RDKit atom object
        Returns: Feature vector as numpy array """

        element_vec = self._one_of_k_encoding_unk(atom.GetSymbol(), PERIODIC_ELEMENTS)
        degree = [atom.GetDegree() / 10.0]
        num_h = [atom.GetTotalNumHs() / 10.0]
        is_aromatic = [float(atom.GetIsAromatic())]
        resname_vec = self._one_of_k_encoding_unk("LIG", list(AA_3TO1.keys()))

        hybrid = atom.GetHybridization().name
        hybridization = self._one_of_k_encoding_unk(hybrid, HYBRIDIZATION)
        features = element_vec + resname_vec + hybridization + degree + num_h + is_aromatic

        return np.array(features)

    def _bond_features_one_hot(self, bond_type: str) -> np.ndarray:
        """Extract features for a bond."""
        return np.array(self._one_of_k_encoding_unk(bond_type,BOND_TYPES_COVALENT + BOND_TYPES_NON_COVALENT))

    def _extract_atom_features(self):
        """Extract one-hot encoded features for each non-hydrogen atom."""

        for i, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                feat = self._atom_features_one_hot(atom)
                #torch.tensor(cont_array, dtype=torch.float)
                self.features.append(torch.tensor(feat, dtype=torch.float))
            else:
                self.index_H_atoms.append(i)


    def _normalize_3d_coord(self, coord: List[float], min_coord: List[float],
                            max_coord: List[float]) -> np.ndarray:
        """Normalize 3D coordinates using min-max normalization."""
        coord = np.array(coord, dtype=float)
        min_coord = np.array(min_coord, dtype=float)
        max_coord = np.array(max_coord, dtype=float)

        range_coord = max_coord - min_coord
        range_coord[range_coord == 0] = 1.0

        return (coord - min_coord) / range_coord
    

    def _add_sdf_coordinates(self, min_coord: List[float], max_coord: List[float]):
        """Add normalized coordinates from SDF conformer."""
        conf = self.mol.GetConformer()
        self.coordinates = []
        features_with_coords = []

        #if self.use_embedding_nodes:
        #    features = self.continuous_features
        #else:
        #    features = self.features

        for i, feat in enumerate(self.features):
            pos = conf.GetAtomPosition(i)
            coords = self._normalize_3d_coord([pos.x, pos.y, pos.z], min_coord, max_coord)
            self.coordinates.append(coords)
            features_with_coords.append(np.concatenate([feat, coords]))

        self.features = features_with_coords


    def _add_edge_distances(self):
        """Add Euclidean distances to edge features without losing non-covalent multi-hot vectors."""
        edge_features_updated = []
        
        for (src_idx, tgt_idx), edge_feat in zip(self.edges, self.edge_features):
            src_coords = self.coordinates[src_idx]
            tgt_coords = self.coordinates[tgt_idx]

            distance = np.linalg.norm(src_coords - tgt_coords)
            
            # edge_feat tiene forma: [Covalente, NC_1, ..., NC_23, Flexibilidad]
            # Queremos insertar la distancia justo antes de la flexibilidad (el último elemento)
            
            # Opción 1: Slicing de listas/arrays
            updated_feat = list(edge_feat[:-1]) + [distance] + [edge_feat[-1]]
            
            edge_features_updated.append(updated_feat)

        self.edge_features = np.array(edge_features_updated, dtype=np.float32)


    def _build_edges_one_hot(self):
        """Build edges with one-hot encoded bond features."""
        #bonds_kekulized = Chem.SDMolSupplier(self.sdf_path.replace("Ligand_SDF", "Ligand_SDF_Kekulized"), sanitize=False, removeHs=True)[0]
        for bond in self.mol.GetBonds():

       # if bonds_kekulized is None:
      #     raise ValueError(f"Invalid SDF file or molecule at: {self.sdf_path.replace("Ligand_SDF", "Ligand_SDF_Kekulized")}")

        #for bond in bonds_kekulized.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Skip hydrogen atoms
            if a not in self.index_H_atoms and b not in self.index_H_atoms:
                if a < self.num_atoms and b < self.num_atoms:
                    self.edges.append([a, b])
                    bond_feat = self._bond_features_one_hot(bond.GetBondType().name)
                    self.edge_features.append(bond_feat / sum(bond_feat))
        assert len(self.edges) == len(self.edge_features), f"Number of Edges: {len(self.edges)}, Number of rows with Edge Features: {len(self.edge_features)}"
        self.num_edges = len(self.edges)


    def _build_graph(self, 
                    min_coord: Optional[List[float]] = None,
                    max_coord: Optional[List[float]] = None) -> bool:
        """
        Build the molecular graph representation.

        Args:
            include_coords: Whether to include 3D coordinates in node features
            use_pdb_coords: Whether to use coordinates from PDB file
            pdb_path: Path to PDB file (required if use_pdb_coords=True)
            ligand_name_conversion: DataFrame for ligand name conversion
            min_coord: Minimum coordinates for normalization
            max_coord: Maximum coordinates for normalization
            include_edge_distances: Whether to include distances in edge features

        Returns:
            True if successful, False otherwise
        """
        # Extract atom features
        try:
            self._extract_atom_features()
            # Add coordinates if requested
            
            if not len(self.features) == len(self.atom_names_pdb):
                print("Error not same features and atom names in the PDB, check it")
            self.num_atoms = len(self.features)

            self._add_sdf_coordinates(min_coord, max_coord)

            # Build edges
            self._build_edges_one_hot()
            self._add_edge_distances()

            return True
        except Exception as e:
            print(f"Error building graph for {self.name}: {e}")
            return False






class LigandCollection:

    def __init__(self):
        self.ligands = {}
        self.failed_ligands = []

    def add_ligand(self, sdf_path, pdb_id, min_coord, max_coord):
        try:
            ligand = Ligand(sdf_path, min_coord, max_coord)
            self.ligands[pdb_id] = ligand
        except Exception as e:
            print(f"Failed to load ligand {pdb_id}: {e}")
            self.failed_ligands.append(pdb_id)

