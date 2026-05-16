---
license: cc-by-sa-4.0
task_categories:
- text-generation
- feature-extraction
language:
- en
tags:
- rdf
- knowledge-graph
- semantic-web
- triples
size_categories:
- 10K<n<100K
---

# ConceptNet 5.7

## Dataset Description

Common-sense knowledge graph with everyday facts and relationships

**Original Source:** https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

### Dataset Summary

This dataset contains RDF triples from ConceptNet 5.7 converted to HuggingFace dataset format
for easy use in machine learning pipelines.

- **Format:** Originally csv, converted to HuggingFace Dataset
- **Size:** 1.2 GB (extracted)
- **Entities:** ~8M nodes
- **Triples:** ~21M edges
- **Original License:** CC BY-SA 4.0

### Recommended Use

Common-sense reasoning, NLP tasks, multilingual knowledge

### Notes\n\nCSV format (tab-separated), requires conversion to RDF. Crowdsourced with confidence scores.


## Dataset Format: Lossless RDF Representation

This dataset uses a **standard lossless format** for representing RDF (Resource Description Framework)
data in HuggingFace Datasets. All semantic information from the original RDF knowledge graph is preserved,
enabling perfect round-trip conversion between RDF and HuggingFace formats.

### Schema

Each RDF triple is represented as a row with **6 fields**:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `subject` | string | Subject of the triple (URI or blank node) | `"http://schema.org/Person"` |
| `predicate` | string | Predicate URI | `"http://www.w3.org/1999/02/22-rdf-syntax-ns#type"` |
| `object` | string | Object of the triple | `"John Doe"` or `"http://schema.org/Thing"` |
| `object_type` | string | Type of object: `"uri"`, `"literal"`, or `"blank_node"` | `"literal"` |
| `object_datatype` | string | XSD datatype URI (for typed literals) | `"http://www.w3.org/2001/XMLSchema#integer"` |
| `object_language` | string | Language tag (for language-tagged literals) | `"en"` |

### Example: RDF Triple Representation

**Original RDF (Turtle)**:
```turtle
<http://example.org/John> <http://schema.org/name> "John Doe"@en .
```

**HuggingFace Dataset Row**:
```python
{
  "subject": "http://example.org/John",
  "predicate": "http://schema.org/name",
  "object": "John Doe",
  "object_type": "literal",
  "object_datatype": None,
  "object_language": "en"
}
```

### Loading the Dataset

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("CleverThis/conceptnet")

# Access the data
data = dataset["data"]

# Iterate over triples
for row in data:
    subject = row["subject"]
    predicate = row["predicate"]
    obj = row["object"]
    obj_type = row["object_type"]

    print(f"Triple: ({subject}, {predicate}, {obj})")
    print(f"  Object type: {obj_type}")
    if row["object_language"]:
        print(f"  Language: {row['object_language']}")
    if row["object_datatype"]:
        print(f"  Datatype: {row['object_datatype']}")
```

### Converting Back to RDF

The dataset can be converted back to any RDF format (Turtle, N-Triples, RDF/XML, etc.) with **zero information loss**:

```python
from datasets import load_dataset
from rdflib import Graph, URIRef, Literal, BNode

def convert_to_rdf(dataset_name, output_file="output.ttl", split="data"):
    """Convert HuggingFace dataset back to RDF Turtle format."""
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Create RDF graph
    graph = Graph()

    # Convert each row to RDF triple
    for row in dataset[split]:
        # Subject
        if row["subject"].startswith("_:"):
            subject = BNode(row["subject"][2:])
        else:
            subject = URIRef(row["subject"])

        # Predicate (always URI)
        predicate = URIRef(row["predicate"])

        # Object (depends on object_type)
        if row["object_type"] == "uri":
            obj = URIRef(row["object"])
        elif row["object_type"] == "blank_node":
            obj = BNode(row["object"][2:])
        elif row["object_type"] == "literal":
            if row["object_datatype"]:
                obj = Literal(row["object"], datatype=URIRef(row["object_datatype"]))
            elif row["object_language"]:
                obj = Literal(row["object"], lang=row["object_language"])
            else:
                obj = Literal(row["object"])

        graph.add((subject, predicate, obj))

    # Serialize to Turtle (or any RDF format)
    graph.serialize(output_file, format="turtle")
    print(f"Exported {len(graph)} triples to {output_file}")
    return graph

# Usage
graph = convert_to_rdf("CleverThis/conceptnet", "reconstructed.ttl")
```

### Information Preservation Guarantee

This format preserves **100% of RDF information**:

- ✅ **URIs**: Exact string representation preserved
- ✅ **Literals**: Full text content preserved
- ✅ **Datatypes**: XSD and custom datatypes preserved (e.g., `xsd:integer`, `xsd:dateTime`)
- ✅ **Language Tags**: BCP 47 language tags preserved (e.g., `@en`, `@fr`, `@ja`)
- ✅ **Blank Nodes**: Node structure preserved (identifiers may change but graph isomorphism maintained)

**Round-trip guarantee**: Original RDF → HuggingFace → Reconstructed RDF produces **semantically identical** graphs.

### Querying the Dataset

You can filter and query the dataset like any HuggingFace dataset:

```python
from datasets import load_dataset

dataset = load_dataset("CleverThis/conceptnet")

# Find all triples with English literals
english_literals = dataset["data"].filter(
    lambda x: x["object_type"] == "literal" and x["object_language"] == "en"
)
print(f"Found {len(english_literals)} English literals")

# Find all rdf:type statements
type_statements = dataset["data"].filter(
    lambda x: "rdf-syntax-ns#type" in x["predicate"]
)
print(f"Found {len(type_statements)} type statements")

# Convert to Pandas for analysis
import pandas as pd
df = dataset["data"].to_pandas()

# Analyze predicate distribution
print(df["predicate"].value_counts())
```

### Dataset Format

The dataset contains all triples in a single **data** split, suitable for machine learning tasks such as:

- Knowledge graph completion
- Link prediction
- Entity embedding
- Relation extraction
- Graph neural networks

### Format Specification

For complete technical documentation of the RDF-to-HuggingFace format, see:

📖 [RDF to HuggingFace Format Specification](https://github.com/CleverThis/cleverernie/blob/master/docs/rdf_huggingface_format_specification.md)

The specification includes:
- Detailed schema definition
- All RDF node type mappings
- Performance benchmarks
- Edge cases and limitations
- Complete code examples

### Conversion Metadata

- **Source Format**: csv
- **Original Size**: 1.2 GB
- **Conversion Tool**: [CleverErnie RDF Pipeline](https://github.com/CleverThis/cleverernie)
- **Format Version**: 1.0
- **Conversion Date**: 2025-11-05


## Citation

If you use this dataset, please cite the original source:

**Original Dataset:** ConceptNet 5.7
**URL:** https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
**License:** CC BY-SA 4.0

## Dataset Preparation

This dataset was prepared using the CleverErnie GISM framework:

```bash
# Download original dataset
cleverernie download-dataset -d conceptnet

# Convert to HuggingFace format
python scripts/convert_rdf_to_hf_dataset.py \
    datasets/conceptnet/[file] \
    hf_datasets/conceptnet \
    --format turtle

# Upload to HuggingFace Hub
python scripts/upload_all_datasets.py --dataset conceptnet
```

## Additional Information

### Original Source

https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz

### Conversion Details

- Converted using: [CleverErnie GISM](https://github.com/cleverthis/cleverernie)
- Conversion script: `scripts/convert_rdf_to_hf_dataset.py`
- Dataset format: Single 'data' split with all triples

### Maintenance

This dataset is maintained by the CleverThis organization.
