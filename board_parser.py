from bs4 import BeautifulSoup
import json

with open("easy_20x20.html", "r") as fd:
    source = fd.read()

tree = BeautifulSoup(source)

cells = []
for cell in tree.find(id="game").div.find_all("div"):
    print(cell)

    if "cell-off" in cell["class"]:
        cells.append(None)
    else:
        clues = [ int(span.text) for span in cell.find_all("span") ]
        cells.append(clues)
    

with open("puzzle.json", "w+") as fd:
    fd.write(json.dumps(cells))
# print(tree.find(id="game").div)

