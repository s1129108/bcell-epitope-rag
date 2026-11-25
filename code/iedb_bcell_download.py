#!/usr/bin/env python3

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

try:
    import requests
except ImportError:
    requests = None

csv.field_size_limit(10_000_000)   # some columns (iri_search) are very long

DEFAULT_BASE_URL = "https://query-api.iedb.org"
ENDPOINT = "bcell_search"
PAGE_SIZE = 5000
MAX_RETRIES = 5
RETRY_BACKOFF = 3.0
REQUEST_TIMEOUT = 180

# ---------------------------------------------------------------------------
# Columns verified with --probe, grouped by what they are used for.
# ---------------------------------------------------------------------------

COLS_CORE = [
    "bcell_id",                 # assay id, used for ordering and pagination
    "structure_id",             # epitope id (several assays can share one structure_id)
    "linear_sequence",
    "linear_sequence_length",
    "qualitative_measure",      # Positive / Positive-High/... / Negative
    "structure_type",           # "Linear peptide"
    "epitope_structure_defined",
    "e_modification",
]

# Metadata used to build the edges of the knowledge graph.
COLS_KG = [
    # source antigen (normalised to UniProt where available)
    "parent_source_antigen_iri",            # UNIPROT:Q4DWE3
    "parent_source_antigen_name",
    "parent_source_antigen_source_org_iri",
    "parent_source_antigen_source_org_name",
    # antigen as declared by the authors + position of the peptide on the protein
    "curated_source_antigen",               # dict: accession, start, end
    # source organism at strain level
    "source_organism_iri",                  # NCBITaxon:353153
    "source_organism_name",
    "source_organism_iri_search",           # the WHOLE taxonomy branch (list)
    # host organism
    "host_organism_iri",
    "host_organism_name",
    # assay
    "assay_iris",
    "assay_names",
    # antibody isotype
    "antibody_isotype",
    # disease
    "disease_iris",
    "disease_names",
    # literature source
    "reference_id",
    "pubmed_id",
    # 3D structure where available
    "pdb_id",
]

DEFAULT_SELECT = COLS_CORE + COLS_KG

# Strict labels: by default only "Positive" and "Negative" are used.
STRICT_POSITIVE_LABELS = {"Positive"}
# Graded positive levels. In IEDB all three mean the epitope WAS recognised; the
# High/Intermediate/Low part describes the strength of the response, not how certain
# the call is. Use --include-graded to merge them into the positive set.
GRADED_POSITIVE_LABELS = {
    "Positive-High", "Positive-Intermediate", "Positive-Low",
}
NEGATIVE_LABELS = {"Negative"}

# Assigned at run time according to the --include-graded flag
POSITIVE_LABELS = set(STRICT_POSITIVE_LABELS)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def build_session():
    if requests is None:
        sys.exit("requests is required:  pip install requests")
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "iedb-bcell-collector/2.0 (research use)",
    })
    return s


def api_get(session, url, params, extra_headers=None, retries=MAX_RETRIES):
    headers = dict(extra_headers or {})
    delay, last_err = RETRY_BACKOFF, None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, params=params, headers=headers,
                            timeout=REQUEST_TIMEOUT)
            if r.status_code in (200, 206):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
            else:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        except requests.RequestException as e:
            last_err = str(e)
        if attempt < retries:
            print(f"  [retry {attempt}/{retries}] {last_err} - waiting {delay:.0f}s",
                  file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Failed after {retries} attempts: {last_err}")


def probe_columns(session, base_url):
    r = api_get(session, f"{base_url}/{ENDPOINT}", {"select": "*", "limit": 1})
    rows = r.json()
    if not rows:
        print("No records returned.")
        return
    print(f"Columns available in '{ENDPOINT}' ({len(rows[0])} columns):\n")
    for k in sorted(rows[0]):
        v = rows[0][k]
        print(f"  {k:<45} {str(v)[:60] if v is not None else '(null)'}")


def get_total_count(session, base_url, filters):
    params = dict(filters, select="bcell_id", limit=1)
    r = api_get(session, f"{base_url}/{ENDPOINT}", params,
                extra_headers={"Prefer": "count=exact"})
    cr = r.headers.get("content-range", "")
    if "/" in cr:
        total = cr.rsplit("/", 1)[-1]
        if total.isdigit():
            return int(total)
    return None


def download_all(session, base_url, out_path, select_cols, filters,
                 page_size=PAGE_SIZE, max_records=None):
    url = f"{base_url}/{ENDPOINT}"
    total = get_total_count(session, base_url, filters)
    print(f"Records matching the filter: {total:,}" if total
          else "Could not read the total count - continuing anyway.")

    offset = written = 0
    writer = fieldnames = None
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        while True:
            if max_records and written >= max_records:
                break
            limit = min(page_size, max_records - written) if max_records else page_size
            params = dict(filters,
                          select=",".join(select_cols) if select_cols else "*",
                          order="bcell_id.asc",   # REQUIRED when paginating
                          limit=limit, offset=offset)
            rows = api_get(session, url, params).json()
            if not rows:
                break
            if writer is None:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
            for row in rows:
                writer.writerow({
                    k: (json.dumps(row.get(k), ensure_ascii=False)
                        if isinstance(row.get(k), (list, dict))
                        else row.get(k))
                    for k in fieldnames
                })
            written += len(rows)
            offset += len(rows)
            pct = f" ({written / total * 100:.1f}%)" if total else ""
            print(f"  downloaded {written:,} records{pct}")
            if len(rows) < limit:
                break
            fh.flush()
    print(f"\nWrote {written:,} records to '{out_path}'")
    return written


# ---------------------------------------------------------------------------
# Reading helpers
# ---------------------------------------------------------------------------

def parse_json_cell(value):
    """list/dict columns are written to CSV as JSON. Returns None when empty."""
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    s = str(value).strip()
    if not s or s.lower() in ("null", "none", "nan"):
        return None
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def as_list(value):
    v = parse_json_cell(value)
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if x not in (None, "")]
    return [v]


def clean_sequence(raw):
    """Return the sequence upper-cased if it is valid, otherwise None."""
    if not raw:
        return None
    seq = str(raw).strip().upper()
    if not seq or not set(seq) <= VALID_AA:
        return None
    return seq


def read_fasta_sequences(path):
    """Read a FASTA file and return the set of sequences, upper-cased."""
    seqs, cur = set(), []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.add("".join(cur).upper())
                    cur = []
            elif line:
                cur.append(line)
    if cur:
        seqs.add("".join(cur).upper())
    return seqs


# ---------------------------------------------------------------------------
# Filtering and splitting into positives / negatives
# ---------------------------------------------------------------------------

def process_dataset(raw_path, out_prefix, min_len=5, max_len=60):
    by_seq = defaultdict(lambda: {"pos": 0, "neg": 0, "other": 0, "ids": []})
    label_counts = defaultdict(int)       # raw distribution of qualitative_measure
    n_read = n_no_seq = n_bad_aa = n_bad_len = 0

    with open(raw_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n_read += 1
            seq = clean_sequence(row.get("linear_sequence"))
            if seq is None:
                if (row.get("linear_sequence") or "").strip():
                    n_bad_aa += 1
                else:
                    n_no_seq += 1
                continue
            if not (min_len <= len(seq) <= max_len):
                n_bad_len += 1
                continue
            qm = (row.get("qualitative_measure") or "").strip()
            label_counts[qm or "(empty)"] += 1
            rec = by_seq[seq]
            rec["ids"].append(row.get("bcell_id", ""))
            if qm in POSITIVE_LABELS:
                rec["pos"] += 1
            elif qm in NEGATIVE_LABELS:
                rec["neg"] += 1
            else:
                rec["other"] += 1

    positives, negatives, conflicts = [], [], []
    for seq, rec in by_seq.items():
        if rec["pos"] and rec["neg"]:
            conflicts.append((seq, rec))
        elif rec["pos"]:
            positives.append((seq, rec))
        elif rec["neg"]:
            negatives.append((seq, rec))

    print("\n--- qualitative_measure distribution (valid records) ---")
    total_lbl = sum(label_counts.values()) or 1
    for lbl, c in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        if lbl in POSITIVE_LABELS:
            tag = "-> positive"
        elif lbl in NEGATIVE_LABELS:
            tag = "-> negative"
        elif lbl in GRADED_POSITIVE_LABELS:
            tag = "-> DROPPED (re-enable with --include-graded)"
        else:
            tag = "-> ignored"
        print(f"  {lbl:<26} {c:>9,}  ({c / total_lbl * 100:5.1f}%)  {tag}")

    n_graded = sum(c for l, c in label_counts.items()
                   if l in GRADED_POSITIVE_LABELS)
    if n_graded and not (POSITIVE_LABELS & GRADED_POSITIVE_LABELS):
        print(f"\n  Dropping {n_graded:,} graded positive records "
              f"({n_graded / total_lbl * 100:.1f}% of the total).")

    print("\n--- Processing statistics ---")
    print(f"Records read:             {n_read:,}")
    print(f"  dropped: no sequence    {n_no_seq:,}")
    print(f"  dropped: invalid chars  {n_bad_aa:,}")
    print(f"  dropped: length range   {n_bad_len:,}")
    print(f"Unique sequences:         {len(by_seq):,}")
    print(f"  positive:               {len(positives):,}")
    print(f"  negative:               {len(negatives):,}")
    print(f"  CONFLICTING (dropped):  {len(conflicts):,}")

    def write_fasta(path, items, prefix):
        with open(path, "w", encoding="utf-8") as fh:
            for i, (seq, rec) in enumerate(sorted(items), 1):
                fh.write(f">{prefix}_{i}|len={len(seq)}|"
                         f"assays={rec['pos'] + rec['neg']}\n{seq}\n")

    write_fasta(f"{out_prefix}_positive.fasta", positives, "Positive_IEDB")
    write_fasta(f"{out_prefix}_negative.fasta", negatives, "Negative_IEDB")

    with open(f"{out_prefix}_conflicts.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence", "n_positive_assays", "n_negative_assays",
                    "bcell_ids"])
        for seq, rec in sorted(conflicts):
            w.writerow([seq, rec["pos"], rec["neg"],
                        ";".join(str(x) for x in rec["ids"][:50])])

    print(f"\nWrote {out_prefix}_positive.fasta, _negative.fasta, "
          f"_conflicts.csv")
    print("The number of conflicting sequences is worth reporting: it is a direct "
          "measure of the label noise in IEDB.")


# ---------------------------------------------------------------------------
# Leakage filtering
# ---------------------------------------------------------------------------

def kmer_set(seq, k):
    return {seq[i:i + k] for i in range(len(seq) - k + 1)} if len(seq) >= k else {seq}


def build_exclusion_filter(exclude_paths, identity_threshold, kmer_size=5):
    """
    Return a function is_excluded(seq) -> bool.

    - Exact match: always excluded.
    - Similarity: approximated by the fraction of shared k-mers (a directed Jaccard
      index). THIS IS AN APPROXIMATION, NOT CD-HIT. Use it as a fast screen; for a
      paper, re-run the filtering with CD-HIT or BLAST and report that number.
    """
    excluded_exact = set()
    for p in exclude_paths:
        excluded_exact |= read_fasta_sequences(p)
    print(f"Loaded {len(excluded_exact):,} sequences to exclude "
          f"from {len(exclude_paths)} file(s)")

    if identity_threshold is None:
        return (lambda s: s in excluded_exact), len(excluded_exact)

    # inverted index: k-mer -> the sequences that contain it
    index = defaultdict(set)
    ex_kmers = {}
    for s in excluded_exact:
        ks = kmer_set(s, kmer_size)
        ex_kmers[s] = ks
        for km in ks:
            index[km].add(s)

    def is_excluded(seq):
        if seq in excluded_exact:
            return True
        qk = kmer_set(seq, kmer_size)
        if not qk:
            return False
        cand = defaultdict(int)
        for km in qk:
            for s in index.get(km, ()):
                cand[s] += 1
        for s, shared in cand.items():
            if shared / min(len(qk), len(ex_kmers[s])) >= identity_threshold:
                return True
        return False

    return is_excluded, len(excluded_exact)


# ---------------------------------------------------------------------------
# Building the knowledge graph
# ---------------------------------------------------------------------------

def build_knowledge_graph(raw_path, out_dir, exclude_paths=None,
                          identity_threshold=None, min_len=5, max_len=60,
                          kmer_size=5, keep_only_path=None):
    os.makedirs(out_dir, exist_ok=True)

    is_excluded = None
    n_excl_src = 0

    if keep_only_path:
        # Whitelist: keep only the sequences present in this file.
        # Use it when the filtering has already been done externally with CD-HIT.
        keep = read_fasta_sequences(keep_only_path)
        print(f"Whitelist: {len(keep):,} sequences from '{keep_only_path}'")
        is_excluded = lambda s: s not in keep
        n_excl_src = len(keep)
    elif exclude_paths:
        is_excluded, n_excl_src = build_exclusion_filter(
            exclude_paths, identity_threshold, kmer_size)

    # node: key -> attributes
    peptides = {}
    antigens, organisms, hosts, assays, isotypes, diseases = ({} for _ in range(6))
    taxon_parent = set()          # (child_iri, parent_iri)
    # edge = (src_type, src_id, relation, dst_type, dst_id, start, end)
    # start/end are used only by the BELONGS_TO relation (position of the peptide
    # on the protein)
    edges = set()

    def add_edge(st, sid, rel, dt, did, start="", end=""):
        edges.add((st, str(sid), rel, dt, str(did), str(start), str(end)))

    n_read = n_skip_seq = n_skip_len = n_leak = 0

    with open(raw_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n_read += 1
            seq = clean_sequence(row.get("linear_sequence"))
            if seq is None:
                n_skip_seq += 1
                continue
            if not (min_len <= len(seq) <= max_len):
                n_skip_len += 1
                continue
            if is_excluded and is_excluded(seq):
                n_leak += 1
                continue

            qm = (row.get("qualitative_measure") or "").strip()
            pep = peptides.setdefault(seq, {
                "sequence": seq, "length": len(seq),
                "n_positive": 0, "n_negative": 0, "n_other": 0,
                "structure_ids": set(),
            })
            if qm in POSITIVE_LABELS:
                pep["n_positive"] += 1
            elif qm in NEGATIVE_LABELS:
                pep["n_negative"] += 1
            else:
                pep["n_other"] += 1
            if row.get("structure_id"):
                pep["structure_ids"].add(str(row["structure_id"]))

            # --- antigen ---
            ag_iri = (row.get("parent_source_antigen_iri") or "").strip()
            ag_name = (row.get("parent_source_antigen_name") or "").strip()
            cur_ag = parse_json_cell(row.get("curated_source_antigen"))
            start = end = None
            if isinstance(cur_ag, dict):
                start = cur_ag.get("starting_position")
                end = cur_ag.get("ending_position")
                if not ag_iri and cur_ag.get("iri"):
                    ag_iri = cur_ag["iri"]
                    ag_name = cur_ag.get("name", ag_name)
            if ag_iri:
                antigens.setdefault(ag_iri, {
                    "antigen_iri": ag_iri, "name": ag_name,
                    "accession": (cur_ag or {}).get("accession", "")
                    if isinstance(cur_ag, dict) else "",
                })
                # The position is an ATTRIBUTE of the edge, not a relation type of
                # its own. Baking the position into the relation name would blow the
                # relation set up to thousands of types and break every KG embedding
                # model.
                add_edge("peptide", seq, "BELONGS_TO", "antigen", ag_iri,
                         "" if start is None else start,
                         "" if end is None else end)

            # --- source organism (strain level) ---
            org_iri = (row.get("source_organism_iri") or "").strip()
            org_name = (row.get("source_organism_name") or "").strip()
            if org_iri:
                organisms.setdefault(org_iri, {"organism_iri": org_iri,
                                               "name": org_name})
                add_edge("peptide", seq, "FROM_ORGANISM", "organism", org_iri)
                if ag_iri:
                    add_edge("antigen", ag_iri, "OF_ORGANISM", "organism", org_iri)

            # --- taxonomy branch: edges for propagation up the taxonomic tree ---
            lineage = [x for x in as_list(row.get("source_organism_iri_search"))
                       if str(x).startswith("NCBITaxon:")]
            for anc in lineage:
                if anc != org_iri:
                    organisms.setdefault(anc, {"organism_iri": anc, "name": ""})
                    taxon_parent.add((org_iri or anc, anc))

            # --- host organism ---
            host_iri = (row.get("host_organism_iri") or "").strip()
            if host_iri:
                hosts.setdefault(host_iri, {
                    "host_iri": host_iri,
                    "name": (row.get("host_organism_name") or "").strip()})
                add_edge("peptide", seq, "TESTED_IN_HOST", "host", host_iri)

            # --- assay ---
            a_iris = as_list(row.get("assay_iris"))
            a_names = [n for n in (row.get("assay_names") or "").split("|") if n]
            for i, a_iri in enumerate(a_iris):
                assays.setdefault(a_iri, {
                    "assay_iri": a_iri,
                    "name": a_names[i] if i < len(a_names) else ""})
                add_edge("peptide", seq, "MEASURED_BY", "assay", a_iri)

            # --- antibody isotype ---
            iso = (row.get("antibody_isotype") or "").strip()
            if iso:
                isotypes.setdefault(iso, {"isotype": iso})
                add_edge("peptide", seq, "RECOGNIZED_BY", "isotype", iso)

            # --- disease ---
            d_iris = as_list(row.get("disease_iris"))
            d_names = as_list(row.get("disease_names"))
            for i, d_iri in enumerate(d_iris):
                diseases.setdefault(d_iri, {
                    "disease_iri": d_iri,
                    "name": d_names[i] if i < len(d_names) else ""})
                add_edge("peptide", seq, "ASSOCIATED_WITH", "disease", d_iri)

    for child, parent in taxon_parent:
        add_edge("organism", child, "CHILD_OF", "organism", parent)

    # ---- write the output files ----
    def dump(name, rows, fields):
        path = os.path.join(out_dir, name)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return len(rows)

    pep_rows = []
    for seq, p in sorted(peptides.items()):
        label = ("conflict" if p["n_positive"] and p["n_negative"]
                 else "positive" if p["n_positive"]
                 else "negative" if p["n_negative"] else "unknown")
        pep_rows.append({**p, "label": label,
                         "structure_ids": ";".join(sorted(p["structure_ids"]))})

    n_pep = dump("nodes_peptide.csv", pep_rows,
                 ["sequence", "length", "label", "n_positive", "n_negative",
                  "n_other", "structure_ids"])
    n_ag = dump("nodes_antigen.csv", sorted(antigens.values(),
                key=lambda d: d["antigen_iri"]),
                ["antigen_iri", "name", "accession"])
    n_org = dump("nodes_organism.csv", sorted(organisms.values(),
                 key=lambda d: d["organism_iri"]), ["organism_iri", "name"])
    n_host = dump("nodes_host.csv", sorted(hosts.values(),
                  key=lambda d: d["host_iri"]), ["host_iri", "name"])
    n_assay = dump("nodes_assay.csv", sorted(assays.values(),
                   key=lambda d: d["assay_iri"]), ["assay_iri", "name"])
    n_iso = dump("nodes_isotype.csv", sorted(isotypes.values(),
                 key=lambda d: d["isotype"]), ["isotype"])
    n_dis = dump("nodes_disease.csv", sorted(diseases.values(),
                 key=lambda d: d["disease_iri"]), ["disease_iri", "name"])

    edge_path = os.path.join(out_dir, "edges.csv")
    with open(edge_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source_type", "source_id", "relation",
                    "target_type", "target_id", "start_pos", "end_pos"])
        for e in sorted(edges):
            w.writerow(e)

    rel_counts = defaultdict(int)
    for e in edges:
        rel_counts[e[2]] += 1

    print("\n--- Knowledge Graph ---")
    print(f"Records read:             {n_read:,}")
    print(f"  dropped: invalid sequence {n_skip_seq:,}")
    print(f"  dropped: length range     {n_skip_len:,}")
    if keep_only_path:
        print(f"  DROPPED (not in whitelist):   {n_leak:,}")
    elif exclude_paths:
        print(f"  DROPPED AS LEAKAGE:       {n_leak:,}  "
              f"(against {n_excl_src:,} train/test sequences)")
    print(f"\nNodes:")
    print(f"  peptide   {n_pep:,}")
    print(f"  antigen   {n_ag:,}")
    print(f"  organism  {n_org:,}")
    print(f"  host      {n_host:,}")
    print(f"  assay     {n_assay:,}")
    print(f"  isotype   {n_iso:,}")
    print(f"  disease   {n_dis:,}")
    print(f"\nEdges: {len(edges):,}")
    for rel, c in sorted(rel_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {rel:<22} {c:,}")
    print(f"\nWrote to directory '{out_dir}/'")

    if (exclude_paths or keep_only_path) and n_pep:
        print(f"\nFor the paper: the KG keeps {n_pep:,} peptides after removing "
              f"{n_leak:,} leaking records.")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Collect B-cell epitopes from IEDB and build a knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="query-api.iedb.org (IEDB) or cedar-api.iedb.org (CEDAR)")
    p.add_argument("--out", default="iedb_bcell_raw.csv")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--count-only", action="store_true")
    p.add_argument("--process", action="store_true")
    p.add_argument("--process-only", action="store_true")
    p.add_argument("--build-kg", action="store_true",
                   help="build the knowledge graph from an existing CSV file")
    p.add_argument("--kg-dir", default="kg_out")
    p.add_argument("--exclude", nargs="*", default=None, metavar="FASTA",
                   help="train/test FASTA files to exclude from the KG")
    p.add_argument("--identity-threshold", type=float, default=None,
                   help="approximate similarity threshold (0-1), e.g. 0.7; "
                        "omit it to exclude exact matches only")
    p.add_argument("--kmer-size", type=int, default=5)
    p.add_argument("--keep-only", default=None, metavar="FASTA",
                   help="keep only the sequences present in this FASTA file (a "
                        "whitelist); use it when the filtering was already done "
                        "externally with CD-HIT. Overrides --exclude.")
    p.add_argument("--all-columns", action="store_true")
    p.add_argument("--page-size", type=int, default=PAGE_SIZE)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--min-len", type=int, default=5)
    p.add_argument("--max-len", type=int, default=60)
    p.add_argument("--include-graded", action="store_true",
                   help="merge Positive-High/Intermediate/Low into the positive set; "
                        "by default ONLY the labels 'Positive' and 'Negative' are used")
    args = p.parse_args()

    global POSITIVE_LABELS
    POSITIVE_LABELS = (STRICT_POSITIVE_LABELS | GRADED_POSITIVE_LABELS
                       if args.include_graded else set(STRICT_POSITIVE_LABELS))
    print(f"Positive labels in use: {sorted(POSITIVE_LABELS)}")
    print(f"Negative labels in use: {sorted(NEGATIVE_LABELS)}\n")

    out_prefix = os.path.splitext(args.out)[0]

    if args.build_kg:
        build_knowledge_graph(args.out, args.kg_dir, args.exclude,
                              args.identity_threshold, args.min_len,
                              args.max_len, args.kmer_size, args.keep_only)
        return
    if args.process_only:
        process_dataset(args.out, out_prefix, args.min_len, args.max_len)
        return

    session = build_session()
    if args.probe:
        probe_columns(session, args.base_url)
        return

    filters = {"linear_sequence": "not.is.null"}
    if args.count_only:
        total = get_total_count(session, args.base_url, filters)
        print(f"Total records: {total:,}" if total else "Could not read the count.")
        return

    download_all(session, args.base_url, args.out,
                 None if args.all_columns else DEFAULT_SELECT, filters,
                 page_size=args.page_size, max_records=args.max_records)
    if args.process:
        process_dataset(args.out, out_prefix, args.min_len, args.max_len)


if __name__ == "__main__":
    main()