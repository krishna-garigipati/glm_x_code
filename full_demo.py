#!/usr/bin/env python
"""KG builder + GLM-X full pipeline demo on a 1000-line corpus."""

import sys, time, json, logging, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("demo")

# ── 1. Generate diverse 1000-line corpus ──────────────────────────────────
def make_corpus(n=1000):
    random.seed(0)

    people = "Albert Einstein|Marie Curie|Isaac Newton|Charles Darwin|Nikola Tesla|Alan Turing|Ada Lovelace|Grace Hopper|Thomas Edison|Alexander Graham Bell|Louis Pasteur|Gregor Mendel|Michael Faraday|Max Planck|Niels Bohr|Erwin Schrodinger|Werner Heisenberg|Enrico Fermi|Rosalind Franklin|James Watson|Francis Crick|Alexander Fleming|Jonas Salk|Edward Jenner|Robert Koch|Galileo Galilei|Dmitri Mendeleev|Stephen Hawking|Richard Feynman".split("|")
    cities = "Paris|London|Berlin|Rome|Madrid|Tokyo|New York|Beijing|Moscow|Sydney|Cairo|Mumbai|Seoul|Toronto|Chicago|San Francisco|Boston|Amsterdam|Vienna|Prague|Barcelona|Lisbon|Dublin|Oslo|Stockholm|Helsinki|Athens|Istanbul|Bangkok|Singapore".split("|")
    countries = "France|England|Germany|Italy|Spain|Japan|United States|China|Russia|Australia|Egypt|India|South Korea|Canada|Brazil|Netherlands|Sweden|Norway|Denmark|Finland|Greece|Turkey|Switzerland|Portugal|Mexico|Argentina|Kenya|South Africa|Nigeria|Israel".split("|")
    rivers = "Seine|Thames|Danube|Rhine|Amazon|Nile|Yangtze|Mississippi|Ganges|Volga|Mekong".split("|")
    animals = "dog|cat|elephant|tiger|lion|bear|wolf|eagle|shark|whale|dolphin|penguin|kangaroo|panda|koala|horse|cow|sheep|goat".split("|")
    techs = "Python|JavaScript|Java|C++|Rust|Go|TypeScript|React|Linux|PostgreSQL|MongoDB|Redis|Docker|Kubernetes|Android|TensorFlow|PyTorch".split("|")

    def pick(lst): return random.choice(lst)
    lines = []

    # lines 1-200: Person VERBed NOUN
    acts = ["developed the theory of relativity","discovered radium","formulated the laws of motion","invented the telephone","created the light bulb","introduced quantum theory","established the uncertainty principle","built the first nuclear reactor","discovered penicillin","developed the polio vaccine"]
    for _ in range(200):
        lines.append(f"{pick(people)} {pick(acts)}.")

    # lines 201-400: X is in Y
    for _ in range(200):
        lines.append(f"{pick(cities)} is in {pick(countries)}.")

    # lines 401-500: River is in Y
    for _ in range(100):
        lines.append(f"The {pick(rivers)} river is in {pick(countries)}.")

    # lines 501-650: X is a Y (definitions)
    for _ in range(150):
        lines.append(f"A {pick(animals)} is an animal.")

    # lines 651-750: X causes Y
    causes = [
        ("Smoking","lung cancer"),("Exercise","good health"),("Pollution","global warming"),
        ("Vaccination","immunity"),("Education","economic growth"),("Stress","heart disease"),
        ("Solar energy","clean electricity"),("Earthquakes","tsunamis"),
        ("Reading","knowledge"),("Gravity","orbital motion"),("Friction","heat"),
        ("Photosynthesis","oxygen production"),("Natural selection","evolution"),
    ]
    for _ in range(100):
        c, e = pick(causes)
        lines.append(f"{c} causes {e}.")

    # lines 751-850: X is part of Y
    parts = [("The CPU","a computer"),("The heart","the circulatory system"),
             ("The liver","the digestive system"),("An engine","a car"),
             ("A chapter","a book"),("A petal","a flower"),("The nucleus","an atom"),
             ("A cell","an organism"),("A wheel","a bicycle"),("A wing","an airplane"),
             ("A continent","the Earth"),("A processor","a smartphone")]
    for _ in range(100):
        p1, p2 = pick(parts)
        lines.append(f"{p1} is part of {p2}.")

    # lines 851-920: X has Y (properties)
    props = [("Water","high heat capacity"),("Metals","electrical conductivity"),
             ("Diamonds","extreme hardness"),("Birds","feathers"),("Fish","gills"),
             ("Mammals","hair"),("Trees","roots"),("Stars","nuclear fusion"),
             ("Computers","memory"),("Plants","chlorophyll"),
             ("Light","wave-particle duality"),("Virus","a protein coat")]
    for _ in range(70):
        s, p = pick(props)
        lines.append(f"{s} has {p}.")

    # lines 921-970: X originated in Y
    origins = [("Coffee","Ethiopia"),("Chocolate","Mesoamerica"),("Paper","China"),
               ("Democracy","Ancient Greece"),("The compass","China"),
               ("Algebra","the Middle East"),("Jazz music","New Orleans"),
               ("Buddhism","India"),("Chess","India"),("Opera","Italy")]
    for _ in range(50):
        s, o = pick(origins)
        lines.append(f"{s} originated in {o}.")

    # lines 971-1000: X is the opposite of Y
    opps = [("hot","cold"),("light","darkness"),("love","hate"),("life","death"),
            ("peace","war"),("wealth","poverty"),("knowledge","ignorance"),
            ("order","chaos"),("truth","lies"),("creation","destruction"),
            ("courage","fear"),("freedom","captivity"),("strength","weakness"),
            ("growth","decay"),("expansion","contraction"),("unity","division"),
            ("clarity","confusion"),("silence","noise"),("summer","winter"),
            ("day","night"),("rise","fall"),("progress","regression")]
    for _ in range(30):
        a, b = pick(opps)
        lines.append(f"{a} is the opposite of {b}.")

    random.shuffle(lines)
    return lines[:n]

corpus = make_corpus(1000)
corpus_path = Path("kg_builder/tests/corpus_1000.txt")
corpus_path.write_text("\n".join(corpus), encoding="utf-8")
print(f"Corpus: {len(corpus)} lines → {corpus_path}")

# ── 2. KG Builder ─────────────────────────────────────────────────────────
print("\n=== [1/4] KG Builder: extracting triples from corpus ===")
from kg_builder import KGBuilderPipeline, KGBuilderConfig
builder = KGBuilderPipeline()
result = builder.process_corpus(corpus, show_progress=True)
stats = result["stats"]
gd = result["graph_data"]
print(f"  Extracted: {stats['raw_triples']} raw → {stats['kept_triples']} kept triples")
print(f"  Graph: {stats['nodes']} nodes, {stats['edges']} edges")
print(f"  Relations: {gd['relation_types']}")
print(f"  Time: {stats['time_seconds']}s")

# ── 3. Build DictGraphStore ───────────────────────────────────────────────
print("\n=== [2/4] Building DictGraphStore ===")
store = builder.build_graph_store(gd)
print(f"  Store: {store.get_node_count()} nodes, {len(store._edges_raw)} edges")
print(f"  Relations: {store.get_all_relations()}")

# ── 4. GLM-X Pipeline ─────────────────────────────────────────────────────
print("\n=== [3/4] Loading GLM-X pipeline ===")
from scripts.glmx_ask import GLMXPipeline
pipeline = GLMXPipeline()
pipeline.graph_store = store
pipeline.load_models()
print("  GLM-X pipeline ready!")

# ── 5. Ask 4 questions from corpus ────────────────────────────────────────
questions = [
    "What is a dog?",
    "What is the opposite of hot?",
    "Is Paris in France?",
    "What causes lung cancer?",
]

print("\n=== [4/4] Asking 4 questions ===")
for q in questions:
    print(f"\n{'='*70}")
    print(f"❓ Q: {q}")
    print(f"{'='*70}")
    result = pipeline.ask(q)
    print(f"✅ A: {result['answer']}")
    print(f"\n   Intent: {result['plan_intents']} ({', '.join(result['plan_names'])})")
    print(f"   Heuristic: {result['heuristic_used']} | Template matched: {result['template_matched']}")
    print(f"   Plan confidence: {result['confidence']:.4f} | Walk confidence: {result['walk_confidence']:.4f}")
    print(f"   Walk steps: {result['n_walk_steps']}")
    print(f"   Path: {' → '.join(result['walk_path_labels'])}")
    print(f"   Edges: {result['walk_path_edges']}")
    for d in result['intent_details']:
        print(f"   {d}")
    print(f"   Subgraph: {result['n_resonated_nodes']} nodes, {result['n_resonated_edges']} edges (energy={result['resonance_energy']:.4f})")
    print(f"   ⏱ {result['time_seconds']}s")

print(f"\n{'='*70}")
print("DEMO COMPLETE")
print(f"{'='*70}")
