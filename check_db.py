import sqlite3
conn = sqlite3.connect('C:\\Users\\obili\\OneDrive\\Documents\\genesis_frameworks\\glm_x_code\\test_results\\tester-a\\datasets\\nature_weather_small.db')
cur = conn.cursor()
cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
print([r[0] for r in cur.fetchall()])
cur.execute('PRAGMA table_info(edges)')
[print(r) for r in cur.fetchall()]
cur.execute('PRAGMA table_info(nodes)')
[print(r) for r in cur.fetchall()]
cur.execute('SELECT * FROM edges')
for row in cur.fetchall():
    print(row)
print('---')
cur.execute('SELECT * FROM nodes')
for row in cur.fetchall():
    print(row)
conn.close()