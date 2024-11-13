import re
import difflib
from typing import Set, Dict, List, Tuple
import os
import math
import traceback

from crystalbleu import corpus_bleu
from collections import Counter
from nltk.util import ngrams
from typing import List, Dict, Any, Union, Optional
# export PNGtoTIKZNPATH="/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# from detikzify.evaluate.crystalbleu import CrystalBLEU
from detikzify.evaluate.eed import TexEditDistance


from collections import Counter
from functools import cached_property
from hashlib import md5
from itertools import chain, tee
from pickle import dump, load
from typing import List

from crystalbleu import corpus_bleu
from datasets.utils.logging import get_logger
from huggingface_hub import cached_assets_path
from pygments.lexers.markup import TexLexer
from pygments.token import Comment, Name, Text
from sacremoses import MosesTokenizer
from torchmetrics import Metric
from datasets import load_dataset
import random
logger = get_logger("datasets")
        
# adopted from nltk
def pad_sequence(sequence, n, pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence

# adopted from nltk
def ngrams(sequence, n, **kwargs):
    sequence = pad_sequence(sequence, n, **kwargs)
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.

def get_datikz_corpus(num_samples: int = 1000) -> List[str]:
    """
    Get TikZ code corpus from nllg/datikz-v2 dataset
    
    Args:
        num_samples: Number of samples to include in corpus
    Returns:
        List of TikZ code strings
    """
    try:
        # Load the dataset
        dataset = load_dataset("nllg/datikz-v2")
        
        # Get the training split
        train_data = dataset['train']
        
        # Extract TikZ codes from 'code' column
        tikz_codes = train_data['code']
        
        # Sample randomly if we have more data than needed
        if len(tikz_codes) > num_samples:
            tikz_codes = random.sample(tikz_codes, num_samples)
        
        # print(f"Loaded {len(tikz_codes)} TikZ code samples for corpus")
        return tikz_codes
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []


class CrystalBLEU(Metric):
    """Wrapper around https://github.com/sola-st/crystalbleu (adapted for LaTeX)"""

    def __init__(self, corpus, k=500, n=4, use_cache=True, **kwargs):
        super().__init__(**kwargs)
        self.lexer = TexLexer()
        self.tokenizer = MosesTokenizer()
        self.use_cache = use_cache
        self.corpus = [c.encode('utf-8') if isinstance(c, str) else c for c in corpus]
        self.k = k
        self.n = n

        self.add_state("list_of_references", [], dist_reduce_fx="cat")
        self.add_state("hypotheses", [], dist_reduce_fx="cat")

    def _tokenize(self, text):
        tokens = []
        for tokentype, value in self.lexer.get_tokens(text):
            if value.strip() and not tokentype is Comment:
                if any(tokentype is tp for tp in [Text, Name.Attribute, Name.Builtin]):
                    tokens.extend(self.tokenizer.tokenize(value.strip()))
                else:
                    tokens.append(value.strip())
        # Convert tokens to bytes
        return [t.encode('utf-8') if isinstance(t, str) else t for t in tokens]

    @cached_property
    def trivially_shared_ngrams(self):
        """Computes trivially shared ngrams and caches them."""
        cache_dir = cached_assets_path(library_name="evaluate", namespace=self.__class__.__name__.lower())
        dhash = md5()
        dhash.update(str(sorted(self.corpus)).encode())
        hashname = f"{dhash.hexdigest()}.pkl"

        if (cache_file:=(cache_dir / hashname)).is_file() and self.use_cache:
            logger.info(f"Found cached trivially shared ngrams ({cache_file})")
            with open(cache_file, "rb") as f:
                return load(f)
        else:
            all_ngrams = []
            for o in range(1, self.n+1):
                for tex in self.corpus:
                    if isinstance(tex, str):
                        tex = tex.encode('utf-8')
                    all_ngrams.extend(ngrams(self._tokenize(tex), o))
            frequencies = Counter(all_ngrams)

            trivially_shared_ngrams = dict(frequencies.most_common(self.k))
            if self.use_cache:
                logger.info(f"Caching trivially shared ngrams ({cache_file})")
                with open(cache_file, "wb") as f:
                    dump(trivially_shared_ngrams, f)
            return trivially_shared_ngrams

    def update(self, list_of_references: List[List[str]], hypotheses: List[str]):
        """Update states with new references and hypotheses."""
        assert len(list_of_references) == len(hypotheses)
        
        # Convert and tokenize references
        processed_refs = []
        for refs in list_of_references:
            ref_group = []
            for ref in refs:
                if isinstance(ref, str):
                    ref_group.append(self._tokenize(ref))
                else:
                    ref_group.append(ref)
            processed_refs.append(ref_group)
        
        # Convert and tokenize hypotheses
        processed_hyps = []
        for hyp in hypotheses:
            if isinstance(hyp, str):
                processed_hyps.append(self._tokenize(hyp))
            else:
                processed_hyps.append(hyp)
        
        self.list_of_references.extend(processed_refs)
        self.hypotheses.extend(processed_hyps)

    def compute(self):
        """Compute CrystalBLEU score."""
        if not self.list_of_references or not self.hypotheses:
            return 0.0
            
        try:
            score = corpus_bleu(
                list_of_references=self.list_of_references,
                hypotheses=self.hypotheses,
                ignoring=self.trivially_shared_ngrams
            )
            # print(f"CrystalBLEU score: {score}")
            return float(score)
        except Exception as e:
            print(f"Error computing CrystalBLEU: {e}")
            return 0.0

class TikZMetrics:
    def __init__(self, corpus: Optional[List[str]] = None, k: int = 500, use_datikz: bool = True):
        """
        Initialize TikZ metrics calculator
        
        Args:
            corpus: Optional list of TikZ code strings
            k: Number of shared n-grams to extract
            use_datikz: Whether to include samples from datikz dataset
        """
        # Get base corpus
        self.corpus = corpus if corpus else []
        
        # Add datikz samples if requested
        if use_datikz:
            datikz_corpus = get_datikz_corpus()
            self.corpus.extend(datikz_corpus)
            # print(f"Total corpus size: {len(self.corpus)} samples")
        
        self.k = k
        
        # Initialize metrics with specific tokenization for TikZ
        self.crystal_bleu = CrystalBLEU(corpus=self.corpus, k=k)
        self.tex_edit_distance = TexEditDistance()
        
        # Define weights
        self.weights = {
            'custom': {
                'command': 0.10,
                'coordinate': 0.10,
                'style': 0.05,
                'sequence': 0.05,
                'edge': 0.10
            },
            'crystal_bleu': 0.35,
            'tex_edit_distance': 0.25
        }

    def preprocess_tikz(self, code: str) -> str:
        """Preprocess TikZ code for better comparison"""
        # Remove comments
        code = re.sub(r'%.*?\n', '\n', code)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        # Normalize coordinates
        code = re.sub(r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)', r'(\1,\2)', code)
        return code.strip()



    def compute_metrics(self, code1: str, code2: str) -> Dict[str, Any]:
        # Preprocess the codes
        code1_clean = self.preprocess_tikz(code1)
        code2_clean = self.preprocess_tikz(code2)
        
        # Get original custom metrics
        custom_metrics = tikz_code_similarity(code1, code2)
        
        # Compute CrystalBLEU score
        self.crystal_bleu.update([[code1]], [code2])  # Pass as list of list for references
        crystal_bleu_score = float(self.crystal_bleu.compute())
        
        # Compute TEX Edit Distance
        self.tex_edit_distance.update([code2_clean], [[code1_clean]])
        ted_score = self.tex_edit_distance.compute()
        ted_normalized = 1.0 - min(ted_score / 100.0, 1.0)

        # Calculate weighted scores
        custom_score = self._calculate_custom_score(custom_metrics)
        
        final_score = (
            custom_score +
            self.weights['crystal_bleu'] * crystal_bleu_score +
            self.weights['tex_edit_distance'] * ted_normalized
        )

        return {
            **custom_metrics,
            'crystal_bleu_score': crystal_bleu_score,
            'tex_edit_distance_raw': ted_score,
            'tex_edit_distance_normalized': ted_normalized,
            'custom_weighted_score': custom_score,
            'crystal_bleu_weighted_score': crystal_bleu_score * self.weights['crystal_bleu'],
            'ted_weighted_score': ted_normalized * self.weights['tex_edit_distance'],
            'final_weighted_score': final_score
        }
    def _calculate_custom_score(self, custom_metrics: Dict[str, Any]) -> float:
        # Track missing elements and weights to redistribute
        missing_elements = custom_metrics["missing_elements"]
        total_weight_to_redistribute = 0.0
        base_weights = self.weights['custom']
        
        # Check each metric and track missing ones
        for key in missing_elements:
            total_weight_to_redistribute += base_weights[key]
        
        # Calculate adjusted weights
        present_elements = [k for k in base_weights.keys() if k not in missing_elements]
        
        if present_elements:
            # Redistribute weights among present elements
            weight_addition = total_weight_to_redistribute / len(present_elements)
            adjusted_weights = {
                k: base_weights[k] + (weight_addition if k in present_elements else 0)
                for k in base_weights
            }
        else:
            # If no elements are present (shouldn't happen but just in case)
            adjusted_weights = {k: 0 for k in base_weights}
            if 'sequence' in base_weights:
                adjusted_weights['sequence'] = 1.0
        
        # Calculate overall score
        overall_score = 0.0
        
        # Add up weighted similarities for present elements
        for key in base_weights.keys():
            similarity_key = f'{key}_similarity'
            if key not in missing_elements and custom_metrics[similarity_key] is not None:
                score_contribution = adjusted_weights[key] * custom_metrics[similarity_key]
                overall_score += score_contribution
                
                # Debug output
                print(f"{key}: {custom_metrics[similarity_key]:.4f} * {adjusted_weights[key]:.4f} = {score_contribution:.4f}")
        
        # # Debug information
        # print("\nCustom Score Analysis:")
        # print("-" * 50)
        # print("Present Elements:", present_elements)
        # print("Missing Elements:", missing_elements)
        # print("\nOriginal Weights:", base_weights)
        # print("Adjusted Weights:", adjusted_weights)
        # print(f"Overall Custom Score: {overall_score:.4f}")
        
        return overall_score
    def adjust_weights(self, 
                      custom_weights: Optional[Dict[str, float]] = None,
                      crystal_bleu_weight: Optional[float] = None,
                      ted_weight: Optional[float] = None):
        """
        Adjust the weights of different metrics. All weights must sum to 1.
        """
        if custom_weights:
            self.weights['custom'] = custom_weights
        if crystal_bleu_weight is not None:
            self.weights['crystal_bleu'] = crystal_bleu_weight
        if ted_weight is not None:
            self.weights['tex_edit_distance'] = ted_weight
            
        # Verify new weights
        custom_sum = sum(self.weights['custom'].values())
        total_sum = custom_sum + self.weights['crystal_bleu'] + self.weights['tex_edit_distance']
        assert abs(total_sum - 1.0) < 1e-6, f"Weights must sum to 1, got {total_sum}"


def tikz_code_similarity(code1: str, code2: str) -> dict:
    def clean_code(code: str) -> str:
        code = re.sub(r'%.*?\n', '\n', code)
        code = re.sub(r'\s+', ' ', code).strip()
        return code
    
    def extract_commands(code: str) -> Set[str]:
        # Basic TikZ commands
        return set(re.findall(r'\\(draw|fill|path|node|circle|rectangle|line|arrow|coordinate)', code))
    
    def extract_edges(code: str) -> Set[str]:
        # Extract edge definitions
        # This includes both explicit \edge commands and -- connections
        edges = set()
        
        # Find explicit edge commands: \edge[style] (node1) -- (node2);
        explicit_edges = re.findall(r'\\edge\s*(?:\[[^\]]*\])?\s*\([^)]*\)\s*--\s*\([^)]*\)', code)
        edges.update(explicit_edges)
        
        # Find direct connections: (coord1) -- (coord2)
        direct_edges = re.findall(r'\([^)]*\)\s*--\s*\([^)]*\)', code)
        edges.update(direct_edges)
        
        # Find to/edge connections in nodes: \node ... to (other_node)
        to_edges = re.findall(r'to\s*(?:\[[^\]]*\])?\s*\([^)]*\)', code)
        edges.update(to_edges)
        
        return edges
    
    def extract_edge_styles(code: str) -> Set[str]:
        # Extract edge-specific styles
        edge_styles = set()
        
        # Find styles in edge commands
        edge_style_blocks = re.findall(r'\\edge\s*\[(.*?)\]', code)
        # Find styles in -- connections
        connection_style_blocks = re.findall(r'--\s*\[(.*?)\]', code)
        # Find styles in to connections
        to_style_blocks = re.findall(r'to\s*\[(.*?)\]', code)
        
        for block in edge_style_blocks + connection_style_blocks + to_style_blocks:
            options = [opt.strip() for opt in block.split(',')]
            edge_styles.update(options)
            
        return edge_styles
    
    def extract_styles(code: str) -> Set[str]:
        styles = []
        style_blocks = re.findall(r'\[(.*?)\]', code)
        for block in style_blocks:
            options = [opt.strip() for opt in block.split(',')]
            styles.extend(options)
        return set(styles)
    
    def extract_coordinates(code: str) -> Set[str]:
        return set(re.findall(r'\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)', code))

    code1_clean = clean_code(code1)
    code2_clean = clean_code(code2)
    
    # Initialize base weights
    base_weights = {
        'command': 0.25,
        'coordinate': 0.25,
        'style': 0.15,
        'sequence': 0.05,
        'edge': 0.30
    }
    
    # Extract all elements
    commands1 = extract_commands(code1_clean)
    commands2 = extract_commands(code2_clean)
    edges1 = extract_edges(code1_clean)
    edges2 = extract_edges(code2_clean)
    coords1 = extract_coordinates(code1_clean)
    coords2 = extract_coordinates(code2_clean)
    styles1 = extract_styles(code1_clean)
    styles2 = extract_styles(code2_clean)
    edge_styles1 = extract_edge_styles(code1_clean)
    edge_styles2 = extract_edge_styles(code2_clean)
    
    # Update styles with edge styles
    styles1.update(edge_styles1)
    styles2.update(edge_styles2)
    
    # Calculate similarities and track missing elements
    similarities = {}
    missing_elements = []
    total_weight_to_redistribute = 0.0
    
    # Command similarity
    if commands1 or commands2:
        similarities['command'] = len(commands1.intersection(commands2)) / max(len(commands1), len(commands2))
    else:
        missing_elements.append('command')
        total_weight_to_redistribute += base_weights['command']
        similarities['command'] = 0
    
    # Edge similarity
    if edges1 or edges2:
        similarities['edge'] = len(edges1.intersection(edges2)) / max(len(edges1), len(edges2))
    else:
        missing_elements.append('edge')
        total_weight_to_redistribute += base_weights['edge']
        similarities['edge'] = None
    
    # Coordinate similarity
    if coords1 or coords2:
        similarities['coordinate'] = len(coords1.intersection(coords2)) / max(len(coords1), len(coords2))
    else:
        missing_elements.append('coordinate')
        total_weight_to_redistribute += base_weights['coordinate']
        similarities['coordinate'] = 0
    
    # Style similarity
    if styles1 or styles2:
        similarities['style'] = len(styles1.intersection(styles2)) / max(len(styles1), len(styles2))
    else:
        missing_elements.append('style')
        total_weight_to_redistribute += base_weights['style']
        similarities['style'] = 0
    
    # Sequence similarity is always calculated
    similarities['sequence'] = difflib.SequenceMatcher(None, code1_clean, code2_clean).ratio()
    
    # Calculate adjusted weights
    present_elements = [k for k in base_weights.keys() if k not in missing_elements]
    if present_elements:
        weight_addition = total_weight_to_redistribute / len(present_elements)
        adjusted_weights = {
            k: base_weights[k] + (weight_addition if k in present_elements else 0)
            for k in base_weights
        }
    else:
        # If no elements are present, give all weight to sequence similarity
        adjusted_weights = {k: 0 for k in base_weights}
        adjusted_weights['sequence'] = 1.0
    
    # Calculate overall similarity
    overall_similarity = sum(
        adjusted_weights[k] * similarities[k]
        for k in similarities
        if k != 'edge' and similarities[k] is not None
    )
    
    # Add edge similarity if present
    if similarities['edge'] is not None:
        overall_similarity += adjusted_weights['edge'] * similarities['edge']
    
    # # Debug information
    # print("\nSimilarity Analysis:")
    # print("-" * 50)
    # print("Present Elements:", present_elements)
    # print("Missing Elements:", missing_elements)
    # print("\nOriginal Weights:", base_weights)
    # print("Adjusted Weights:", adjusted_weights)
    # print("\nSimilarity Scores:")
    # for k, v in similarities.items():
    #     if v is not None:
    #         print(f"{k}: {v:.4f} * {adjusted_weights[k]:.4f} = {v * adjusted_weights[k]:.4f}")
    # print(f"Overall Similarity: {overall_similarity:.4f}")

    return {
        'command_similarity': similarities['command'],
        'coordinate_similarity': similarities['coordinate'],
        'style_similarity': similarities['style'],
        'sequence_similarity': similarities['sequence'],
        'edge_similarity': similarities['edge'],
        'overall_similarity': overall_similarity,
        # Debug information
        'commands1': commands1,
        'commands2': commands2,
        'edges1': edges1,
        'edges2': edges2,
        'styles1': styles1,
        'styles2': styles2,
        'coordinates1': coords1,
        'coordinates2': coords2,
        'adjusted_weights': adjusted_weights,
        'missing_elements': missing_elements
    }


if __name__ == "__main__":
    # Example paths - you can modify these
    original_path = "/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ/difficulty_measure/tikz_results_20241109_152045/images/example_0/combination_1.tex"
    # generated_path = "/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ/difficulty_measure/tikz_results_20241109_152045/images/example_0/combination_1.tex"

    generated_path = "/home/sama.hadhoud/Documents/AI701/project/ours/PNG--to-TIKZ/difficulty_measure/tikz_results_20241109_152045/generated/example_0/generated_1.tex"
    
    # Read the TikZ code files
    try:
        with open(original_path, 'r') as f:
            original_code = f.read()
        with open(generated_path, 'r') as f:
            generated_code = f.read()


        # Initialize with default weights
        metrics_calculator = TikZMetrics(corpus=[original_code, generated_code])

        # print("\nDebug Tokenization:")
        # print("-" * 50)
        # metrics_calculator.crystal_bleu.debug_tokens(original_code)
        # metrics_calculator.crystal_bleu.debug_tokens(generated_code)
                    
        
        # # Or adjust weights if needed
        # metrics_calculator.adjust_weights(
        #     custom_weights={
        #         'command': 0.15,    # 15%
        #         'coordinate': 0.15, # 15%
        #         'style': 0.05,     # 5%
        #         'sequence': 0.02,   # 2%
        #         'edge': 0.13       # 13%
        #     },
        #     crystal_bleu_weight=0.30,  # 30%
        #     ted_weight=0.20           # 20%
        # )
        
        # Compute metrics
        all_metrics = metrics_calculator.compute_metrics(original_code, generated_code)
        
        # Print results
        print("\nWeighted Scores:")
        print("-" * 50)
        print(f"Custom Metrics Score: {all_metrics['custom_weighted_score']:.4f}")
        print(f"CrystalBLEU Score: {all_metrics['crystal_bleu_weighted_score']:.4f}")
        print(f"TED Score: {all_metrics['ted_weighted_score']:.4f}")
        print(f"Final Combined Score: {all_metrics['final_weighted_score']:.4f}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")