print("im trying")
import pandas as pd
print("im trying pandas")
import torch
import esm
import logging
print("im trying logging")
import sys

print("im starting")

# ------------------- Setup Logging -------------------
log_file = "log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ErrorCode %(code)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
def log_error(message, code=999):
    logging.getLogger().info(message, extra={"code": code})

# ------------------- Mutation Function -------------------
def apply_mutation(wt_seq, mutation_code):
    try:
        from_aa = mutation_code[0]
        to_aa = mutation_code[-1]
        pos = int(mutation_code[1:-1]) - 1
        if wt_seq[pos] != from_aa:
            raise ValueError(f"Expected {from_aa} at position {pos+1}, but found {wt_seq[pos]}")
        return wt_seq[:pos] + to_aa + wt_seq[pos+1:]
    except Exception as e:
        log_error(f"Error applying mutation {mutation_code}: {e}", code=101)
        raise

# ------------------- Load ESM Model -------------------
try:
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.cuda()
    model.eval()
    log_error("ESM-2 model loaded successfully", code=0)
except Exception as e:
    log_error(f"Failed to load ESM-2 model: {e}", code=102)
    sys.exit(1)

# ------------------- Embedding Function -------------------
def get_embedding(seq):
    try:
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", seq)])
        batch_tokens = batch_tokens.to(device='cuda')
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[6])
        token_representations = results["representations"][6].cpu()
        return token_representations[0, 1:-1].mean(0).numpy()
    except Exception as e:
        log_error(f"Error embedding sequence: {seq[:10]}... : {e}", code=103)
        return None

# ------------------- Load Excel -------------------
try:
    xls = pd.ExcelFile("DMS_failed.xlsx") 
    sheet_dfs = {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    log_error("Excel file loaded successfully", code=0)
except Exception as e:
    log_error(f"Error loading Excel file: {e}", code=104)
    sys.exit(1)

# ------------------- Process Sheets -------------------
for sheet_name, df in sheet_dfs.items():
    try:
        print(f"\nProcessing sheet: {sheet_name}")
        if not {'variant', 'fitness_scaled'}.issubset(df.columns):
            log_error(f"Skipping {sheet_name}: missing required columns", code=105)
            continue

        original = df['Original_seq'].dropna().iloc[0]
        df = df[['variant', 'fitness_scaled']].dropna()

        # Apply mutation
        df['sequences'] = df['variant'].apply(lambda v: apply_mutation(original, v))

        # Embedding
        print(f"Embedding {len(df)} sequences...")
        embeddings = []
        for seq in df['sequences']:
            emb = get_embedding(seq)
            if emb is not None:
                embeddings.append(emb)
            else:
                log_error(f"Skipping sequence due to embedding failure: {seq[:10]}...", code=106)

        if len(embeddings) != len(df):
            log_error(f"{len(df) - len(embeddings)} sequences failed to embed", code=107)

        embed_df = pd.DataFrame(embeddings)
        embed_df['fitness_scaled'] = df['fitness_scaled'].values[:len(embeddings)]

        out_file = f"{sheet_name}_esm2_embeddings.csv"
        embed_df.to_csv(out_file, index=False)
        print(f"Saved embeddings to: {out_file}")
        log_error(f"Finished processing sheet: {sheet_name}", code=0)

    except Exception as e:
        log_error(f"Fatal error in sheet {sheet_name}: {e}", code=108)
