import sqlite3

c = sqlite3.connect("test_results/stage_a/toy_graph.db")
print("nodes in table:", c.execute("select count(*) from nodes").fetchone()[0])
print("ids 1..12:")
for r in c.execute("select id,label from nodes where id between 1 and 12 order by id"):
    print("   ", r)
print("label like %dog%:", c.execute("select id,label from nodes where label like ?", ("%dog%",)).fetchall())
print("id 7 ->", c.execute("select id,label from nodes where id=7").fetchall())
print("edges touching 14 (tail):", c.execute(
    "select source_id,target_id,relation from edges where source_id=14 or target_id=14").fetchall())
print("edges touching 7:", c.execute(
    "select source_id,target_id,relation from edges where source_id=7 or target_id=7").fetchall())
print("max node id:", c.execute("select max(id) from nodes").fetchone()[0])
print("nodes table sql:", c.execute(
    "select sql from sqlite_master where name='nodes'").fetchone()[0])