# data_pipeline/nextpandemics_ligand.py

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from data_pipeline.atom_utils import HBD_PATTERN, HBA_PATTERN, FLEXIBILITY_BOND_PATTERN

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

    def __init__(self, sdf_path: str, min_coord: list, max_coord: list):
        """
        Initialize a Ligand object from an SDF file.

        Args:
            sdf_path: Path to the SDF file
        """
        self.sdf_path = sdf_path
        self.mol = None
        self.name = None
        self.features = []
        self.edges = []
        self.edge_features = []
        self.coordinates = []
        self.atom_names_pdb = None
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
        self._build_graph(min_coord, max_coords) 


    def _extract_atom_names_from_cif(self):
        """
        Read a CIF file and extract atom names from the _atom_site.label_atom_id column.
        Returns a list of atom names (e.g., ['N', 'C', 'C1', 'N1', ...]).
        """
        cif_path = self.sdf_path.replace("Ligand_SDF", "Ligand_CIF").replace(".sdf", ".cif")
        atom_names = []
        if self.sdf_path.split('\\')[-1].split('_')[0] == '9G0I':
            d = 0
        with open(cif_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("HETATM", "ATOM")):
                    fields = line.split()
                    if len(fields) >= 4:  # ensure column exists
                       if not fields[3].startswith('H') and (fields[4] == '.' or fields[4] == 'A'):
                            atom_names.append(fields[3])
        return atom_names

    def _load_molecule(self):
        """Load molecule from SDF file."""

        mol_supplier = Chem.SDMolSupplier(self.sdf_path, sanitize=True, removeHs=False)

        self.mol = mol_supplier[0]

        if self.mol is None:
            raise ValueError(f"Invalid SDF file or molecule at: {self.sdf_path}")

        # Extract name
        self.name = self.mol.GetProp("_Name") if self.mol.HasProp("_Name") else "UNK"
        self.atom_names_pdb = self._extract_atom_names_from_cif()
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
            self._build_edges(simplified_edge_distances=simplified_edge_distances)

            # Add distances to edges if requested
            if include_edge_distances:
                self._add_edge_distances()
                # Remove coords from node features if not explicitly requested
                if not include_coords:
                    self.features = [f[:-3] for f in self.features]
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

