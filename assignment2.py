import os
import random
import time
from collections import defaultdict

# ────────────────────────────────────────────────
#  Helper functions
# ────────────────────────────────────────────────

def jaccard_sim(set1, set2):
    if not set1 and not set2:
        return 1.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0


def get_char_kgrams(text, k):
    grams = set()
    for i in range(len(text) - k + 1):
        grams.add(text[i:i+k])
    return grams


def get_word_kgrams(text, k):
    words = text.split()
    grams = set()
    for i in range(len(words) - k + 1):
        grams.add(' '.join(words[i:i+k]))
    return grams


# ────────────────────────────────────────────────
#  1. Load documents (fixed path)
# ────────────────────────────────────────────────

docs = {}
try:
    for i in range(1, 5):
        path = f"minhash/D{i}.txt"
        with open(path, 'r', encoding='utf-8') as f:
            docs[i] = f.read().strip()
    print("All 4 documents loaded successfully.\n")
except FileNotFoundError as e:
    print("ERROR: Could not find one or more files.")
    print("Make sure you have a folder called 'minhash' in the same directory as this script.")
    print("And that it contains: D1.txt  D2.txt  D3.txt  D4.txt")
    print(f"Exact error: {e}")
    exit(1)

# ────────────────────────────────────────────────
#  1. Create k-grams & exact Jaccard (Part 1B)
# ────────────────────────────────────────────────

print("=== Part 1 ===")
print("Exact Jaccard similarities (all pairs, 3 types)")

char2 = {i: get_char_kgrams(docs[i], 2) for i in range(1,5)}
char3 = {i: get_char_kgrams(docs[i], 3) for i in range(1,5)}
word2 = {i: get_word_kgrams(docs[i], 2) for i in range(1,5)}

gram_sets = {
    "char-2grams": char2,
    "char-3grams": char3,
    "word-2grams": word2
}

for name, dic in gram_sets.items():
    print(f"\n{name}:")
    for i in range(1,4):
        for j in range(i+1,5):
            sim = jaccard_sim(dic[i], dic[j])
            print(f"  D{i} ↔ D{j}: {sim:.4f}")

# Save exact D1-D2 char-3gram for later comparison
exact_d1_d2 = jaccard_sim(char3[1], char3[2])
print(f"\nExact Jaccard (D1-D2, char-3grams): {exact_d1_d2:.4f}\n")

# ────────────────────────────────────────────────
#  MinHash helper functions
# ────────────────────────────────────────────────

# We use two different large prime moduli as requested in different parts
P_LARGE   = 2**61 - 1           # very large → almost no collision
P_MEDIUM  = 10007               # > 10,000 as suggested in part 2

def generate_hash_functions(t, modulus):
    random.seed(42)  # for reproducibility in one run
    return [(random.randint(1, modulus-1), random.randint(0, modulus-1)) for _ in range(t)]


def minhash_signature(shingles, hash_functions, modulus):
    sig = []
    for a, b in hash_functions:
        if not shingles:
            sig.append(0)
            continue
        minval = min((a * x + b) % modulus for x in shingles)
        sig.append(minval)
    return sig


def approx_jaccard(sig1, sig2):
    if len(sig1) != len(sig2):
        return 0.0
    matches = sum(1 for a,b in zip(sig1,sig2) if a == b)
    return matches / len(sig1)


# Convert char-3grams to integers (base-27 encoding)
def shingle_to_int(s):
    val = 0
    for c in s:
        if c == ' ':
            i = 26
        else:
            i = ord(c) - ord('a')
        val = val * 27 + i
    return val


char3_int = {i: {shingle_to_int(g) for g in char3[i]} for i in range(1,5)}

# ────────────────────────────────────────────────
#  Part 1A: MinHash D1-D2 with large modulus (as in original part 1)
# ────────────────────────────────────────────────

print("=== Part 1A ===")
print("MinHash approximation D1 ↔ D2 (char-3grams, large modulus)")

ts = [20, 60, 150, 300, 600]

for t in ts:
    start = time.time()
    hf = generate_hash_functions(t, P_LARGE)
    sig1 = minhash_signature(char3_int[1], hf, P_LARGE)
    sig2 = minhash_signature(char3_int[2], hf, P_LARGE)
    est = approx_jaccard(sig1, sig2)
    print(f"t = {t:3d}  →  est J = {est:.4f}   (time: {time.time()-start:.3f}s)")

# ────────────────────────────────────────────────
#  Part 2: MinHash with m > 10,000 (medium modulus)
# ────────────────────────────────────────────────

print("\n=== Part 2A ===")
print("MinHash approximation D1 ↔ D2 (m ≈ 10007)")

for t in ts:
    start = time.time()
    hf = generate_hash_functions(t, P_MEDIUM)
    sig1 = minhash_signature(char3_int[1], hf, P_MEDIUM)
    sig2 = minhash_signature(char3_int[2], hf, P_MEDIUM)
    est = approx_jaccard(sig1, sig2)
    print(f"t = {t:3d}  →  est J = {est:.4f}   (time: {time.time()-start:.3f}s)")

# ────────────────────────────────────────────────
#  Part 2B: Which t is good? (example experiment)
# ────────────────────────────────────────────────

print("\n=== Part 2B – experiments for good t ===")
print(f"Exact = {exact_d1_d2:.4f}\n")

for t in [20, 50, 100, 150, 300, 500, 800]:
    hf = generate_hash_functions(t, P_MEDIUM)
    sig1 = minhash_signature(char3_int[1], hf, P_MEDIUM)
    sig2 = minhash_signature(char3_int[2], hf, P_MEDIUM)
    est = approx_jaccard(sig1, sig2)
    err = abs(est - exact_d1_d2)
    print(f"t={t:4d}   est={est:.4f}   error={err:.4f}")

print("\nSuggestion: t=150–300 often gives good balance between accuracy and time.")

# ────────────────────────────────────────────────
#  You can continue adding Part 3, 4, 5 later
#  For now this should run without errors and produce Part 1 & 2 output
# ────────────────────────────────────────────────
# ... previous code ends here ...

# ────────────────────────────────────────────────
#  Part 3: LSH
# ────────────────────────────────────────────────

print("\n=== Part 3A: Best r, b for t=160, tau=0.7 ===")

def s_curve(s, b, r):
    return 1 - (1 - s ** b) ** r

t_lsh = 160
tau = 0.7

best_r = 1
best_b = t_lsh
best_separation = -float('inf')

for r in range(1, t_lsh + 1):
    if t_lsh % r != 0:
        continue
    b = t_lsh // r
    f_high = s_curve(tau + 0.1, b, r)
    f_low = s_curve(tau - 0.1, b, r)
    separation = f_high - f_low
    if separation > best_separation:
        best_separation = separation
        best_r = r
        best_b = b

print(f"Best r (bands) = {best_r}, b (hashes/band) = {best_b}")
print(f"  S-curve at s=0.6: {s_curve(0.6, best_b, best_r):.4f}")
print(f"  S-curve at s=0.8: {s_curve(0.8, best_b, best_r):.4f}")
print(f"  Separation: {best_separation:.4f} (good if >0.5; steep curve at tau)")

print("\n=== Part 3B: Prob each pair >0.7 (using best r,b) ===")

pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
for p in pairs:
    s = jaccard_sim(char3[p[0]], char3[p[1]])
    prob = s_curve(s, best_b, best_r)
    print(f"D{p[0]} ↔ D{p[1]} (exact s={s:.4f}): Prob = {prob:.4f}")

# ← no extra spaces here 
# ────────────────────────────────────────────────
#  Part 4 & 5 – MovieLens 100k
# ────────────────────────────────────────────────

print("\n" + "="*50)
print("=== Loading MovieLens 100k data ===")
print("="*50)

ml_path = "ml-100k/u.data"

if not os.path.exists(ml_path):
    print("\nERROR: Cannot find ml-100k/u.data")
    print("Please:")
    print("1. Download ml-100k.zip from https://grouplens.org/datasets/movielens/100k/")
    print("2. Unzip it")
    print("3. Move the whole 'ml-100k' folder next to this script")
    print("   Final path should be: ml-100k/u.data")
    exit(1)

user_movies = defaultdict(set)
with open(ml_path, encoding='latin-1') as f:
    for line in f:
        uid, mid, _, _ = line.strip().split('\t')
        user_movies[int(uid)].add(int(mid))

users = sorted(user_movies.keys())
n_users = len(users)

print(f"→ Loaded {n_users} users with their rated movies.\n")

# ────────────────────────────────────────────────
# Compute exact pairs with Jaccard ≥ 0.5  (takes 10–60 seconds)
# ────────────────────────────────────────────────

print("Computing exact Jaccard similarities ≥ 0.5 ... (please wait)")
start_exact = time.time()

true_pairs_05 = []
exact_sim_dict = {}   # only store if needed — can be memory heavy

for i in range(n_users):
    for j in range(i+1, n_users):
        u1 = users[i]
        u2 = users[j]
        sim = jaccard_sim(user_movies[u1], user_movies[u2])
        if sim >= 0.5:
            true_pairs_05.append((min(u1,u2), max(u1,u2)))
        # Optional: store all sims if you need them later
        # exact_sim_dict[(min(u1,u2), max(u1,u2))] = sim

print(f"→ Found {len(true_pairs_05)} pairs with exact Jaccard ≥ 0.5")
print(f"→ Time: {time.time() - start_exact:.1f} seconds\n")

# ────────────────────────────────────────────────
# MinHash helper functions for MovieLens
# ────────────────────────────────────────────────

P_MOVIE = 10**9 + 7   # large prime ~2 billion

def get_user_signatures(t, seed=42):
    random.seed(seed)
    hash_funcs = generate_hash_functions(t, P_MOVIE)
    signatures = {}
    for u in users:
        movies = user_movies[u]
        sig = minhash_signature(movies, hash_funcs, P_MOVIE)
        signatures[u] = sig
    return signatures

# ────────────────────────────────────────────────
# Part 4 – Min-Hashing
# ────────────────────────────────────────────────

print("="*50)
print("=== Part 4: Min-Hashing on MovieLens ===")
print("="*50)

ts_movie = [50, 100, 200]

for t in ts_movie:
    print(f"\n→ Using t = {t} hash functions  (average over 5 runs)")
    fps = []
    fns = []
    for run in range(5):
        print(f"   Run {run+1}/5 ...", end=" ", flush=True)
        sigs = get_user_signatures(t, seed=42 + run)
        
        # Compute approximate pairs ≥ 0.5
        approx_pairs = []
        for i in range(n_users):
            for j in range(i+1, n_users):
                est = approx_jaccard(sigs[users[i]], sigs[users[j]])
                if est >= 0.5:
                    approx_pairs.append((min(users[i],users[j]), max(users[i],users[j])))
        
        approx_set = set(approx_pairs)
        true_set   = set(true_pairs_05)
        
        fp = len(approx_set - true_set)
        fn = len(true_set - approx_set)
        fps.append(fp)
        fns.append(fn)
        print(f"done (FP={fp}, FN={fn})")
    
    avg_fp = sum(fps) / 5
    avg_fn = sum(fns) / 5
    print(f"   → Average over 5 runs:")
    print(f"      False Positives : {avg_fp:.1f}")
    print(f"      False Negatives : {avg_fn:.1f}\n")

# ────────────────────────────────────────────────
# Part 5 – LSH
# ────────────────────────────────────────────────

def get_lsh_candidates(sigs, r, b):
    candidates = set()
    for band_idx in range(b):
        buckets = defaultdict(list)
        start = band_idx * r
        for u in users:
            band_key = tuple(sigs[u][start : start + r])
            buckets[band_key].append(u)
        for bucket in buckets.values():
            if len(bucket) >= 2:
                bucket_sorted = sorted(bucket)
                for p in range(len(bucket_sorted)):
                    for q in range(p+1, len(bucket_sorted)):
                        candidates.add((bucket_sorted[p], bucket_sorted[q]))
    return list(candidates)

print("="*50)
print("=== Part 5: LSH on MovieLens ===")
print("="*50)

# True pairs at different thresholds
true_06 = [p for p in true_pairs_05 if exact_sim_dict.get(p, 0) >= 0.6]
true_08 = [p for p in true_pairs_05 if exact_sim_dict.get(p, 0) >= 0.8]

print(f"True pairs with Jaccard ≥ 0.6 : {len(true_06)}")
print(f"True pairs with Jaccard ≥ 0.8 : {len(true_08)}\n")

configs = [
    (50,  [(5, 10)]),
    (100, [(5, 20)]),
    (200, [(5, 40), (10, 20)])
]

for t, config_list in configs:
    print(f"Signatures built with {t} hash functions:")
    for r, b in config_list:
        print(f"  → r = {r}, b = {b}")
        fp6s, fn6s, fp8s, fn8s = [], [], [], []
        for run in range(5):
            print(f"     Run {run+1}/5 ...", end=" ", flush=True)
            sigs = get_user_signatures(t, seed=42 + run)
            cands = get_lsh_candidates(sigs, r, b)
            cand_set = set(cands)
            
            fp6 = len(cand_set - set(true_06))
            fn6 = len(set(true_06) - cand_set)
            fp8 = len(cand_set - set(true_08))
            fn8 = len(set(true_08) - cand_set)
            
            fp6s.append(fp6)
            fn6s.append(fn6)
            fp8s.append(fp8)
            fn8s.append(fn8)
            print(f"done")
        
        print(f"     ≥ 0.6 → Avg FP: {sum(fp6s)/5:.1f}   Avg FN: {sum(fn6s)/5:.1f}")
        print(f"     ≥ 0.8 → Avg FP: {sum(fp8s)/5:.1f}   Avg FN: {sum(fn8s)/5:.1f}\n")

print("\n" + "="*50)
print("All parts (1–5) completed!")
print("="*50)
