from g2p.config import G2PConfig
config = G2PConfig.from_yaml('configs/config_g2p.yaml')
config.validate()
print('Config valid!')
print('Similarity threshold:', config.extraction.similarity_threshold)
print('Clause split:', config.extraction.clause_split)
print('Follows variants:', config.extraction.relation_variants['follows'])
print('Precedes variants:', config.extraction.relation_variants['precedes'])
print('Causes variants count:', len(config.extraction.relation_variants['causes']))