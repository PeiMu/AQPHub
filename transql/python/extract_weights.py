"""
extract_weights.py — Download Llama3-8B weights from HuggingFace and save as .npy files.

Usage:
    # PyTorch path (default): precompute RoPE manually, build DAG from C++ hardcode
    python extract_weights.py --output-dir /path/to/weights_npy

    # ONNX path: constant-fold ONNX graph, extract weights + topology.json
    python extract_weights.py --source onnx --onnx-path model.onnx \
        --output-dir /path/to/weights_npy

Outputs (one .npy per weight tensor):
    embed_tokens.npy         [vocab_size, hidden_dim]
    lm_head.npy              [vocab_size, hidden_dim]
    final_norm.npy           [hidden_dim]
    layer_{l}_q_proj.npy     [4096, 4096]
    layer_{l}_k_proj.npy     [1024, 4096]
    layer_{l}_v_proj.npy     [1024, 4096]
    layer_{l}_o_proj.npy     [4096, 4096]
    layer_{l}_gate_proj.npy  [14336, 4096]
    layer_{l}_up_proj.npy    [14336, 4096]
    layer_{l}_down_proj.npy  [4096, 14336]
    layer_{l}_norm1.npy      [4096]    (input_layernorm)
    layer_{l}_norm2.npy      [4096]    (post_attention_layernorm)
    rope_cos.npy             [max_seq_len, num_chunks, chunk_size//2]  (PyTorch path)
    rope_sin.npy             [max_seq_len, num_chunks, chunk_size//2]  (PyTorch path)
    topology.json            DAG node sequence                          (ONNX path only)
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig


MODEL_ID = "meta-llama/Meta-Llama-3-8B"
CHUNK_SIZE = 32
MAX_SEQ_LEN = 2048


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def precompute_rope(config, max_seq_len: int, chunk_size: int):
    """
    Precompute RoPE cos/sin tables in split-chunk format (PyTorch path equivalent
    of the constant folding that onnxsim performs on the ONNX path).

    Returns two arrays shaped [max_seq_len, num_chunks, chunk_size//2]:
        cos_table[pos, chunk_id, i] = cos(pos / 10000^(2*(chunk_id*chunk_size/2 + i) / head_dim))
        sin_table[pos, chunk_id, i] = sin(...)

    head_dim = hidden_size // num_attention_heads = 128
    Each head spans head_dim dims = 4 chunks of 32.
    Dims 2i and 2i+1 in the head share the same angle (standard RoPE).
    We store cos[i] = angle for pair i within the chunk.
    """
    head_dim = config.hidden_size // config.num_attention_heads  # 128
    hidden_dim = config.hidden_size  # 4096
    num_chunks = hidden_dim // chunk_size  # 128 chunks per token

    half = chunk_size // 2  # 16 pairs per chunk

    cos_table = np.zeros((max_seq_len, num_chunks, half), dtype=np.float32)
    sin_table = np.zeros((max_seq_len, num_chunks, half), dtype=np.float32)

    for c in range(num_chunks):
        for i in range(half):
            global_dim = c * chunk_size + 2 * i
            d = global_dim % head_dim  # dim index within head (0..127)
            pair_idx = d // 2          # pair index (0..63)
            theta = 1.0 / (10000.0 ** (2.0 * pair_idx / head_dim))
            for pos in range(max_seq_len):
                cos_table[pos, c, i] = np.cos(pos * theta)
                sin_table[pos, c, i] = np.sin(pos * theta)

    return cos_table, sin_table


def _save_pytorch_weights(config, sd: dict, output_dir: str):
    """Save per-layer and global weights from a PyTorch state_dict."""
    np.save(os.path.join(output_dir, "embed_tokens.npy"),
            sd["model.embed_tokens.weight"])
    np.save(os.path.join(output_dir, "final_norm.npy"),
            sd["model.norm.weight"])
    lm_head_key = "lm_head.weight"
    if lm_head_key not in sd:
        lm_head_key = "model.embed_tokens.weight"
    np.save(os.path.join(output_dir, "lm_head.npy"), sd[lm_head_key])

    num_layers = config.num_hidden_layers
    for l in range(num_layers):
        prefix = f"model.layers.{l}"
        mapping = {
            f"layer_{l}_q_proj":    f"{prefix}.self_attn.q_proj.weight",
            f"layer_{l}_k_proj":    f"{prefix}.self_attn.k_proj.weight",
            f"layer_{l}_v_proj":    f"{prefix}.self_attn.v_proj.weight",
            f"layer_{l}_o_proj":    f"{prefix}.self_attn.o_proj.weight",
            f"layer_{l}_gate_proj": f"{prefix}.mlp.gate_proj.weight",
            f"layer_{l}_up_proj":   f"{prefix}.mlp.up_proj.weight",
            f"layer_{l}_down_proj": f"{prefix}.mlp.down_proj.weight",
            f"layer_{l}_norm1":     f"{prefix}.input_layernorm.weight",
            f"layer_{l}_norm2":     f"{prefix}.post_attention_layernorm.weight",
        }
        for out_name, sd_key in mapping.items():
            np.save(os.path.join(output_dir, out_name + ".npy"), sd[sd_key])
        if (l + 1) % 8 == 0:
            print(f"  Saved layer {l + 1}/{num_layers}")


# ---------------------------------------------------------------------------
# PyTorch path
# ---------------------------------------------------------------------------

def extract_pytorch(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading config for {MODEL_ID}...")
    config = AutoConfig.from_pretrained(MODEL_ID)
    print(f"  rms_norm_eps        = {config.rms_norm_eps}")
    print(f"  num_attention_heads = {config.num_attention_heads}")
    print(f"  num_key_value_heads = {config.num_key_value_heads}")
    print(f"  hidden_size         = {config.hidden_size}")
    print(f"  intermediate_size   = {config.intermediate_size}")
    print(f"  num_hidden_layers   = {config.num_hidden_layers}")

    print(f"\nLoading model weights (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="cpu"
    )
    sd = {k: v.numpy() for k, v in model.state_dict().items()}

    _save_pytorch_weights(config, sd, output_dir)

    # RoPE tables (constant folding equivalent for the PyTorch path)
    print("\nPrecomputing RoPE tables (constant folding equivalent)...")
    cos_table, sin_table = precompute_rope(config, MAX_SEQ_LEN, CHUNK_SIZE)
    np.save(os.path.join(output_dir, "rope_cos.npy"), cos_table)
    np.save(os.path.join(output_dir, "rope_sin.npy"), sin_table)

    print(f"\nDone. Weights saved to {output_dir}")


# ---------------------------------------------------------------------------
# ONNX path
# ---------------------------------------------------------------------------

# Maps ONNX initializer name patterns to canonical TranSQL+ table names.
# Adjust if the exported ONNX uses different naming conventions.
_ONNX_NAME_MAP = {
    "model.embed_tokens.weight":      "embed_tokens",
    "model.norm.weight":              "final_norm",
    "lm_head.weight":                 "lm_head",
}
for _l in range(32):
    _p = f"model.layers.{_l}"
    _ONNX_NAME_MAP.update({
        f"{_p}.self_attn.q_proj.weight":            f"layer_{_l}_q_proj",
        f"{_p}.self_attn.k_proj.weight":            f"layer_{_l}_k_proj",
        f"{_p}.self_attn.v_proj.weight":            f"layer_{_l}_v_proj",
        f"{_p}.self_attn.o_proj.weight":            f"layer_{_l}_o_proj",
        f"{_p}.mlp.gate_proj.weight":               f"layer_{_l}_gate_proj",
        f"{_p}.mlp.up_proj.weight":                 f"layer_{_l}_up_proj",
        f"{_p}.mlp.down_proj.weight":               f"layer_{_l}_down_proj",
        f"{_p}.input_layernorm.weight":             f"layer_{_l}_norm1",
        f"{_p}.post_attention_layernorm.weight":    f"layer_{_l}_norm2",
    })


def _is_matmul_weight(name: str) -> bool:
    """Return True for 2-D weight initializers used as the right-hand operand
    of MatMul (these need transposing from ONNX [in_dim, out_dim] to
    our [out_dim, in_dim] chunking convention)."""
    matmul_suffixes = (
        "_q_proj", "_k_proj", "_v_proj", "_o_proj",
        "_gate_proj", "_up_proj", "_down_proj", "lm_head",
    )
    return any(name.endswith(s) for s in matmul_suffixes)


def _build_topology_json(onnx_graph, output_dir: str, chunk_size: int):
    """
    Walk the (already constant-folded) ONNX graph and produce topology.json.

    This is a best-effort heuristic mapping based on known Llama3-8B ONNX
    structure after onnxsim simplification.  The resulting JSON is consumed
    by TensorComputeDAG::BuildFromJSON in C++.
    """
    import onnx
    from onnx import numpy_helper

    initializer_names = {init.name for init in onnx_graph.initializer}

    # Map ONNX output name → canonical table name (for connecting nodes).
    # We walk ops in topological order (ONNX guarantees this).
    value_to_table: dict[str, str] = {"": "input_tokens"}

    nodes_json = []
    node_id = 0

    # Counters for naming intermediate tensors per layer.
    layer_counters: dict[int, dict] = {}

    def layer_of(name: str) -> int:
        """Infer layer index from an ONNX tensor name, or -1 if global."""
        import re
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else -1

    def get_or_default(d, key, default):
        return d[key] if key in d else default

    # We emit one high-level DAG node per major ONNX node cluster.
    # For simplicity, detect op patterns by node op_type + input/output names.
    for onnx_node in onnx_graph.node:
        op = onnx_node.op_type
        inputs  = list(onnx_node.input)
        outputs = list(onnx_node.output)

        # Skip constant nodes (eliminated by onnxsim but may still appear)
        if op == "Constant":
            continue

        # Gather non-initializer inputs as table names
        input_tables = [
            value_to_table.get(inp, inp)
            for inp in inputs
            if inp and inp not in initializer_names
        ]
        # Gather weight initializer names (canonicalised)
        weight_tables = [
            _ONNX_NAME_MAP.get(inp, inp)
            for inp in inputs
            if inp in initializer_names
        ]

        # Heuristic op-type detection
        dag_op = None
        params: dict[str, str] = {}

        if op == "Gather":
            dag_op = "EmbedLookup"
            # inputs[0] = embedding weight, inputs[1] = indices
            input_tables = ["input_tokens",
                            _ONNX_NAME_MAP.get(inputs[0], inputs[0])]

        elif op == "MatMul":
            dag_op = "MatMul"

        elif op in ("RMSNormalization", "SimplifiedLayerNormalization"):
            dag_op = "RMSNorm"
            params = {"hidden_dim": "4096", "eps": "1e-05",
                      "chunk_size": str(chunk_size)}

        elif op == "Softmax":
            dag_op = "Softmax"

        elif op == "Add":
            dag_op = "ResidualAdd"

        # Skip unrecognised ops (they are subsumed into neighbouring SQL templates)
        if dag_op is None:
            for out in outputs:
                if input_tables:
                    value_to_table[out] = input_tables[0]
            continue

        out_name = f"dag_node_{node_id}"
        for out in outputs:
            value_to_table[out] = out_name

        all_inputs = input_tables + weight_tables
        nodes_json.append({
            "id":            node_id,
            "op_type":       dag_op,
            "output_table":  out_name,
            "input_tables":  all_inputs,
            "is_shared":     False,
            "params":        params,
        })
        node_id += 1

    if nodes_json:
        nodes_json[-1]["is_shared"] = False

    topology = {
        "nodes":          nodes_json,
        "output_node_id": node_id - 1,
    }
    out_path = os.path.join(output_dir, "topology.json")
    with open(out_path, "w") as f:
        json.dump(topology, f, indent=2)
    print(f"  Wrote topology.json ({len(nodes_json)} nodes) → {out_path}")


def extract_onnx(onnx_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    try:
        import onnx
        import onnxsim
    except ImportError:
        raise ImportError(
            "ONNX path requires: pip install onnx onnxsim\n"
            "Also ensure the ONNX model was exported with:\n"
            "  optimum-cli export onnx --model meta-llama/Meta-Llama-3-8B ."
        )
    from onnx import numpy_helper

    print(f"Loading ONNX model from {onnx_path}...")
    model_onnx = onnx.load(onnx_path)
    print(f"  Opset: {model_onnx.opset_import[0].version}, "
          f"Nodes before simplification: {len(model_onnx.graph.node)}")

    # Constant folding: eliminates constant sub-graphs (RoPE cos/sin computation,
    # mask generation, etc.) leaving only parametric ops.
    print("Applying constant folding via onnxsim.simplify()...")
    model_simplified, ok = onnxsim.simplify(model_onnx)
    if not ok:
        print("  Warning: onnxsim could not fully simplify the model; continuing.")
    else:
        model_onnx = model_simplified
    print(f"  Nodes after simplification: {len(model_onnx.graph.node)}")

    # ------------------------------------------------------------------
    # Extract weight initializers
    # ------------------------------------------------------------------
    print("\nExtracting weight initializers...")
    saved = 0
    for init in model_onnx.graph.initializer:
        canonical = _ONNX_NAME_MAP.get(init.name)
        if canonical is None:
            continue  # not a weight we track

        arr = numpy_helper.to_array(init).astype(np.float32)

        # ONNX stores MatMul right-hand operand as [in_dim, out_dim].
        # Our chunking convention needs [out_dim, in_dim].  Transpose.
        if _is_matmul_weight(canonical) and arr.ndim == 2:
            arr = arr.T

        np.save(os.path.join(output_dir, canonical + ".npy"), arr)
        saved += 1

    print(f"  Saved {saved} weight tensors.")

    # ------------------------------------------------------------------
    # Build topology.json
    # ------------------------------------------------------------------
    print("\nBuilding topology.json from ONNX graph...")
    _build_topology_json(model_onnx.graph, output_dir, CHUNK_SIZE)

    print(f"\nDone. Outputs saved to {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write .npy files (and topology.json on ONNX path)")
    parser.add_argument("--source", choices=["pytorch", "onnx"], default="pytorch",
                        help="Weight source: 'pytorch' (default) or 'onnx'")
    parser.add_argument("--onnx-path", default=None,
                        help="Path to exported ONNX model file (required when --source onnx)")
    args = parser.parse_args()

    if args.source == "onnx":
        if not args.onnx_path:
            parser.error("--onnx-path is required when --source onnx")
        extract_onnx(args.onnx_path, args.output_dir)
    else:
        extract_pytorch(args.output_dir)
