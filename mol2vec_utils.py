"""
Mol2vec Utilities for Molecular Property Prediction

This module implements Mol2vec functionality for generating molecular embeddings
from SMILES strings. Mol2vec is an unsupervised machine learning approach that
learns vector representations of molecular substructures using Word2Vec.

Reference:
    Jaeger et al. "Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition"
    J. Chem. Inf. Model. 2018, 58, 1, 27-35
    DOI: 10.1021/acs.jcim.7b00616

Usage:
    from mol2vec_utils import Mol2VecEmbedder
    
    embedder = Mol2VecEmbedder()
    embeddings = embedder.get_embeddings(smiles_list)
"""

import os
import pickle
import warnings
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from gensim.models import Word2Vec
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================================
# Core Mol2vec Functions
# ============================================================================

def mol_to_sentence(mol: Chem.Mol, radius: int = 1) -> List[str]:
    """
    Convert a molecule to a sentence of Morgan fingerprint identifiers.
    
    Each atom generates identifiers at different radii (0 to radius).
    Identifiers are ordered by atom index in canonical SMILES.
    
    Args:
        mol: RDKit molecule object
        radius: Maximum radius for Morgan fingerprint identifiers
        
    Returns:
        List of identifier strings representing the molecular "sentence"
    """
    if mol is None:
        return []
    
    sentence = []
    info = {}
    
    # Generate Morgan fingerprint with bit info
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    # Sort by atom index to ensure consistent ordering
    atom_indices = sorted(set(atom_idx for bit_id in info for atom_idx, r in info[bit_id]))
    
    # For each atom, add identifiers from radius 0 to max radius
    for atom_idx in atom_indices:
        for r in range(radius + 1):
            # Find identifiers at this radius for this atom
            for bit_id, atom_info in info.items():
                for a_idx, rad in atom_info:
                    if a_idx == atom_idx and rad == r:
                        sentence.append(str(bit_id))
    
    return sentence


def smiles_to_sentence(smiles: str, radius: int = 1) -> List[str]:
    """
    Convert a SMILES string to a molecular sentence.
    
    Args:
        smiles: SMILES string
        radius: Maximum radius for Morgan fingerprint identifiers
        
    Returns:
        List of identifier strings representing the molecular "sentence"
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_sentence(mol, radius)


def generate_corpus(
    smiles_list: List[str],
    radius: int = 1,
    n_jobs: int = 1,
    uncommon_threshold: int = 3,
    replace_uncommon: str = "UNK"
) -> Tuple[List[List[str]], dict]:
    """
    Generate a corpus of molecular sentences from SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Maximum radius for Morgan fingerprint identifiers
        n_jobs: Number of parallel jobs (currently not implemented, reserved)
        uncommon_threshold: Identifiers appearing <= this many times are replaced
        replace_uncommon: String to replace uncommon identifiers
        
    Returns:
        Tuple of (corpus, identifier_counts) where corpus is list of sentences
    """
    corpus = []
    identifier_counts = {}
    
    print(f"Generating corpus from {len(smiles_list)} molecules...")
    
    # First pass: generate sentences and count identifiers
    for smiles in tqdm(smiles_list, desc="Parsing molecules"):
        sentence = smiles_to_sentence(smiles, radius)
        if sentence:  # Only add non-empty sentences
            corpus.append(sentence)
            for identifier in sentence:
                identifier_counts[identifier] = identifier_counts.get(identifier, 0) + 1
    
    # Second pass: replace uncommon identifiers
    if replace_uncommon:
        processed_corpus = []
        for sentence in tqdm(corpus, desc="Replacing uncommon identifiers"):
            processed_sentence = [
                identifier if identifier_counts.get(identifier, 0) > uncommon_threshold 
                else replace_uncommon
                for identifier in sentence
            ]
            processed_corpus.append(processed_sentence)
        corpus = processed_corpus
        
        # Add UNK to counts for reference
        identifier_counts[replace_uncommon] = sum(
            1 for s in corpus for w in s if w == replace_uncommon
        )
    
    print(f"Corpus generated: {len(corpus)} sentences")
    print(f"Unique identifiers: {len(identifier_counts)}")
    
    return corpus, identifier_counts


# ============================================================================
# Model Training and Loading
# ============================================================================

def train_mol2vec_model(
    corpus: List[List[str]],
    vector_size: int = 300,
    window: int = 10,
    min_count: int = 1,
    workers: int = 4,
    sg: int = 1,  # 1 for skip-gram, 0 for CBOW
    epochs: int = 5,
    seed: int = 42
) -> Word2Vec:
    """
    Train a Mol2vec (Word2Vec) model on a molecular corpus.
    
    Args:
        corpus: List of molecular sentences (from generate_corpus)
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Minimum word frequency to include
        workers: Number of worker threads
        sg: Training algorithm: 1 for skip-gram, 0 for CBOW
        epochs: Number of training epochs
        seed: Random seed for reproducibility
        
    Returns:
        Trained Word2Vec model
    """
    print(f"Training Mol2vec model...")
    print(f"  - Vector size: {vector_size}")
    print(f"  - Window: {window}")
    print(f"  - Algorithm: {'Skip-gram' if sg == 1 else 'CBOW'}")
    print(f"  - Epochs: {epochs}")
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
        seed=seed
    )
    
    print(f"Model trained! Vocabulary size: {len(model.wv)}")
    
    return model


def save_model(model: Word2Vec, filepath: str):
    """Save Mol2vec model to disk."""
    model.save(filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Word2Vec:
    """Load Mol2vec model from disk."""
    model = Word2Vec.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


# ============================================================================
# Featurization
# ============================================================================

def sentence_to_vector(
    sentence: List[str],
    model: Word2Vec,
    unseen_vec: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a molecular sentence to a vector by averaging word vectors.
    
    Args:
        sentence: List of identifier strings
        model: Trained Word2Vec model
        unseen_vec: Vector to use for unseen identifiers (default: zero vector)
        
    Returns:
        Numpy array of shape (vector_size,)
    """
    vector_size = model.wv.vector_size
    
    if unseen_vec is None:
        # Try to use UNK vector if available
        if "UNK" in model.wv:
            unseen_vec = model.wv["UNK"]
        else:
            unseen_vec = np.zeros(vector_size)
    
    vectors = []
    for word in sentence:
        if word in model.wv:
            vectors.append(model.wv[word])
        else:
            vectors.append(unseen_vec)
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)


def smiles_to_vector(
    smiles: str,
    model: Word2Vec,
    radius: int = 1,
    unseen_vec: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert a SMILES string to a molecular embedding vector.
    
    Args:
        smiles: SMILES string
        model: Trained Word2Vec model
        radius: Radius used during corpus generation
        unseen_vec: Vector to use for unseen identifiers
        
    Returns:
        Numpy array of shape (vector_size,)
    """
    sentence = smiles_to_sentence(smiles, radius)
    return sentence_to_vector(sentence, model, unseen_vec)


def featurize_molecules(
    smiles_list: List[str],
    model: Word2Vec,
    radius: int = 1,
    unseen_vec: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Featurize a list of molecules using Mol2vec.
    
    Args:
        smiles_list: List of SMILES strings
        model: Trained Word2Vec model
        radius: Radius used during corpus generation
        unseen_vec: Vector to use for unseen identifiers
        
    Returns:
        Numpy array of shape (n_molecules, vector_size)
    """
    embeddings = []
    
    for smiles in tqdm(smiles_list, desc="Featurizing molecules"):
        vec = smiles_to_vector(smiles, model, radius, unseen_vec)
        embeddings.append(vec)
    
    return np.array(embeddings)


# ============================================================================
# High-Level Embedder Class
# ============================================================================

class Mol2VecEmbedder:
    """
    High-level class for generating Mol2vec embeddings.
    
    Handles model training, saving/loading, caching, and featurization.
    
    Example:
        embedder = Mol2VecEmbedder()
        
        # Option 1: Train a new model
        embedder.train(smiles_list, save_path="mol2vec_model.pkl")
        
        # Option 2: Load pre-trained model
        embedder.load("mol2vec_model.pkl")
        
        # Get embeddings
        embeddings = embedder.get_embeddings(smiles_list)
    """
    
    def __init__(
        self,
        radius: int = 1,
        vector_size: int = 300,
        window: int = 10,
        cache_file: str = "mol2vec_cache.pt"
    ):
        """
        Initialize the Mol2vec embedder.
        
        Args:
            radius: Maximum radius for Morgan fingerprint identifiers
            vector_size: Dimensionality of embedding vectors
            window: Context window size for Word2Vec
            cache_file: Path to cache file for storing computed embeddings
        """
        self.radius = radius
        self.vector_size = vector_size
        self.window = window
        self.cache_file = cache_file
        self.model = None
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        if os.path.exists(self.cache_file):
            try:
                import torch
                self._cache = torch.load(self.cache_file)
                print(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                print(f"Could not load cache: {e}")
                self._cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            import torch
            torch.save(self._cache, self.cache_file)
        except Exception as e:
            print(f"Could not save cache: {e}")
    
    def train(
        self,
        smiles_list: List[str],
        save_path: Optional[str] = None,
        uncommon_threshold: int = 3,
        epochs: int = 5,
        workers: int = 4
    ) -> "Mol2VecEmbedder":
        """
        Train a Mol2vec model on the given SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings for training
            save_path: Path to save the trained model (optional)
            uncommon_threshold: Threshold for replacing uncommon identifiers
            epochs: Number of training epochs
            workers: Number of worker threads
            
        Returns:
            self (for method chaining)
        """
        # Generate corpus
        corpus, _ = generate_corpus(
            smiles_list,
            radius=self.radius,
            uncommon_threshold=uncommon_threshold,
            replace_uncommon="UNK"
        )
        
        # Train model
        self.model = train_mol2vec_model(
            corpus,
            vector_size=self.vector_size,
            window=self.window,
            epochs=epochs,
            workers=workers
        )
        
        # Save model if path provided
        if save_path:
            save_model(self.model, save_path)
        
        # Clear cache since we have a new model
        self._cache = {}
        self._save_cache()
        
        return self
    
    def load(self, model_path: str) -> "Mol2VecEmbedder":
        """
        Load a pre-trained Mol2vec model.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            self (for method chaining)
        """
        self.model = load_model(model_path)
        self.vector_size = self.model.wv.vector_size
        return self
    
    def get_embeddings(
        self,
        smiles_list: List[str],
        use_cache: bool = True,
        return_tensor: bool = True
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Get Mol2vec embeddings for a list of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            use_cache: Whether to use cached embeddings
            return_tensor: If True, return PyTorch tensor; else numpy array
            
        Returns:
            Embeddings array of shape (n_molecules, vector_size)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call train() or load() first.")
        
        embeddings = []
        to_compute = []
        compute_indices = []
        
        # Check cache
        for i, smiles in enumerate(smiles_list):
            if use_cache and smiles in self._cache:
                embeddings.append(self._cache[smiles])
            else:
                embeddings.append(None)
                to_compute.append(smiles)
                compute_indices.append(i)
        
        # Compute missing embeddings
        if to_compute:
            print(f"Computing {len(to_compute)} new embeddings...")
            new_embeddings = featurize_molecules(to_compute, self.model, self.radius)
            
            for idx, smiles, emb in zip(compute_indices, to_compute, new_embeddings):
                embeddings[idx] = emb
                if use_cache:
                    self._cache[smiles] = emb
            
            if use_cache:
                self._save_cache()
        else:
            print("All embeddings loaded from cache.")
        
        # Stack embeddings
        result = np.array(embeddings)
        
        if return_tensor:
            import torch
            return torch.tensor(result, dtype=torch.float32)
        
        return result
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("Cache cleared.")


# ============================================================================
# Visualization Helpers
# ============================================================================

def visualize_substructure(mol: Chem.Mol, identifier: int, radius: int = 1):
    """
    Visualize the substructure corresponding to a Morgan fingerprint identifier.
    
    Args:
        mol: RDKit molecule object
        identifier: Morgan fingerprint identifier (bit)
        radius: Radius at which the identifier was generated
        
    Returns:
        RDKit drawing or None if identifier not found in molecule
    """
    from rdkit.Chem import Draw
    
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    
    if identifier not in info:
        print(f"Identifier {identifier} not found in molecule")
        return None
    
    # Get atom and radius for this identifier
    atom_idx, r = info[identifier][0]
    
    # Create environment for highlighting
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    
    return Draw.MolToImage(mol, highlightAtoms=list(amap.keys()))


def plot_embedding_tsne(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create t-SNE visualization of molecular embeddings.
    
    Args:
        embeddings: Array of shape (n_molecules, vector_size)
        labels: Optional labels for coloring points
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
        figsize: Figure size
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    print("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    proj = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=figsize)
    
    if labels is not None:
        scatter = plt.scatter(proj[:, 0], proj[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
    else:
        plt.scatter(proj[:, 0], proj[:, 1], alpha=0.7)
    
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Mol2vec Embeddings - t-SNE Projection")
    plt.tight_layout()
    plt.show()
    
    return proj


# ============================================================================
# Integration with Existing Workflow
# ============================================================================

def get_mol2vec_embeddings_for_df(
    df: pd.DataFrame,
    chromophore_col: str = "Chromophore",
    solvent_col: str = "Solvent",
    model_path: Optional[str] = None,
    train_if_missing: bool = True,
    vector_size: int = 300,
    cache_file: str = "mol2vec_embedding_cache.pt"
) -> Tuple[np.ndarray, np.ndarray, "Mol2VecEmbedder"]:
    """
    Generate Mol2vec embeddings for chromophores and solvents from a DataFrame.
    
    This function is designed to integrate with your existing workflow.
    
    Args:
        df: DataFrame with chromophore and solvent SMILES columns
        chromophore_col: Name of chromophore SMILES column
        solvent_col: Name of solvent SMILES column
        model_path: Path to pre-trained model (optional)
        train_if_missing: If True and no model exists, train on the data
        vector_size: Embedding dimension if training new model
        cache_file: Path for embedding cache
        
    Returns:
        Tuple of (chromophore_embeddings, solvent_embeddings, embedder)
        
    Example:
        chrom_emb, solv_emb, embedder = get_mol2vec_embeddings_for_df(
            df, 
            model_path="mol2vec_model.pkl"
        )
        combined_emb = np.concatenate([chrom_emb, solv_emb], axis=1)
    """
    import torch
    
    embedder = Mol2VecEmbedder(
        vector_size=vector_size,
        cache_file=cache_file
    )
    
    # Load or train model
    if model_path and os.path.exists(model_path):
        embedder.load(model_path)
    elif train_if_missing:
        print("Training Mol2vec model on dataset...")
        # Combine all SMILES for training
        all_smiles = list(set(
            df[chromophore_col].dropna().tolist() + 
            df[solvent_col].dropna().tolist()
        ))
        embedder.train(all_smiles, save_path=model_path or "mol2vec_model.pkl")
    else:
        raise ValueError("No model found and train_if_missing=False")
    
    # Get embeddings
    print("\nGenerating chromophore embeddings...")
    chrom_emb = embedder.get_embeddings(
        df[chromophore_col].tolist(),
        return_tensor=False
    )
    
    print("\nGenerating solvent embeddings...")
    solv_emb = embedder.get_embeddings(
        df[solvent_col].tolist(),
        return_tensor=False
    )
    
    return chrom_emb, solv_emb, embedder


# ============================================================================
# Pre-trained Model Download (Optional)
# ============================================================================

def download_pretrained_model(
    model_url: str = None,
    save_path: str = "mol2vec_pretrained.pkl"
) -> str:
    """
    Download a pre-trained Mol2vec model.
    
    Note: You'll need to provide a URL to a pre-trained model.
    The original Mol2vec model can be found in the mol2vec GitHub repository.
    
    Args:
        model_url: URL to download model from
        save_path: Local path to save the model
        
    Returns:
        Path to the saved model
    """
    if model_url is None:
        print("Note: The original pre-trained Mol2vec model is available at:")
        print("https://github.com/samoturk/mol2vec/tree/master/examples/models")
        print("\nYou can also train your own model using the train() method.")
        return None
    
    import urllib.request
    
    print(f"Downloading pre-trained model from {model_url}...")
    urllib.request.urlretrieve(model_url, save_path)
    print(f"Model saved to {save_path}")
    
    return save_path


# ============================================================================
# Main Example
# ============================================================================

if __name__ == "__main__":
    # Example usage
    example_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)O",  # Isopropanol
        "CCCCCCCC",  # Octane
    ]
    
    print("=" * 60)
    print("Mol2vec Example")
    print("=" * 60)
    
    # Create embedder
    embedder = Mol2VecEmbedder(vector_size=100)
    
    # Train on example data
    embedder.train(example_smiles, save_path="example_mol2vec.pkl", epochs=3)
    
    # Get embeddings
    embeddings = embedder.get_embeddings(example_smiles, return_tensor=False)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Example embedding (ethanol): {embeddings[0][:5]}...")
    
    # Show sentences
    print("\nMolecular sentences:")
    for smiles in example_smiles[:3]:
        sentence = smiles_to_sentence(smiles, radius=1)
        print(f"  {smiles}: {sentence}")




