import sys

import torch
import transformers

from typing import NamedTuple, List, Dict, Set, Self, Tuple, IO, Callable


def highlighted(s: str) -> str:
    return "\033[46m" + s + "\033[0m"


class Cell(NamedTuple):
    x: int
    y: int
    letter: str

    def __str__(self):
        return self.letter


class Word(NamedTuple):
    cells: Tuple[Cell, ...]

    def __str__(self):
        return "".join([cell.letter for cell in self.cells])


class Grid:
    grid: List[List[Cell]]

    def __init__(self, letter_grid: List[List[str]]):
        self.grid = [
            [Cell(nx, ny, letter) for ny, letter in enumerate(letter_row)]
            for nx, letter_row
            in enumerate(letter_grid)
        ]

    def get_adj(self, x, y) -> List[Cell]:
        offsets = [(1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
        adj = []
        for x_off, y_off in offsets:
            if 0 <= x_off + x < len(self.grid):
                if 0 <= y_off + y < len(self.grid[x_off + x]):
                    adj.append(self.grid[x + x_off][y_off + y])

        return adj

    def cells(self) -> List[Cell]:
        return [cell for row in self.grid for cell in row]

    def str_highlighted(self, word: Word):
        return "".join([
            " ".join([highlighted(str(cell)) if (cell in word.cells) else str(cell) for cell in row]) + "\n"
            for row
            in self.grid
        ])

    def __str__(self):
        return "".join([
            " ".join([str(cell) for cell in row]) + "\n"
            for row
            in self.grid
        ])


class PrefixTreeNode():
    def __init__(self,
                 is_valid: bool = False,
                 children: Dict[str, Self] = None
                 ):
        if children is None:
            children = {}
        self.is_valid = is_valid
        self.children = children

    def insert(self, item: str):
        if len(item) == 1:
            if item not in self.children.keys():
                self.children[item] = PrefixTreeNode()
            self.children[item] = PrefixTreeNode(True, self.children[item].children)
        else:
            if item[0] not in self.children.keys():
                self.children[item[0]] = PrefixTreeNode()
            self.children[item[0]].insert(item[1:])

    def __contains__(self, item: str) -> bool:
        if len(item) == 0:
            return self.is_valid

        if item[0] in self.children.keys():
            return item[1:] in self.children[item[0]]
        else:
            return False

    def contains_prefix(self, item: str) -> bool:
        if len(item) == 0:
            return True

        if item[0] in self.children.keys():
            return self.children[item[0]].contains_prefix(item[1:])
        else:
            return False

    def __repr__(self) -> str:
        return f"X{self.children}" if self.is_valid else str(self.children)


def find_words_from_cell(cell: Cell, grid: Grid, words: PrefixTreeNode, visited: Word = Word(())) -> Set[Word]:
    visited = Word(visited.cells + (cell,))

    found_words = set()

    if not words.contains_prefix(str(visited)):
        return found_words

    if str(visited) in words and visited != "":
        found_words.add(visited)

    for adj_cell in grid.get_adj(cell.x, cell.y):
        if adj_cell not in visited.cells:
            found_words = found_words.union(find_words_from_cell(adj_cell, grid, words, visited))

    return found_words


def find_words(grid: Grid, words: PrefixTreeNode) -> Set[Word]:
    found_words = set()
    for cell in grid.cells():
        found_words = found_words.union(find_words_from_cell(cell, grid, words))
        # print("X")
    return found_words


def grid_from_file(file: IO[str]) -> Grid:
    lines = [line for line in file]
    letter_grid = [
        [letter.upper() for letter in line if
         ascii("A") <= ascii(letter) <= ascii("Z") or ascii("a") <= ascii(letter) <= ascii("z")]
        for line in lines
    ]
    return Grid(letter_grid)


def word_tree_from_file(file: IO[str]) -> PrefixTreeNode:
    tree = PrefixTreeNode()
    for line in file:
        tree.insert(line.strip().upper())
    return tree


def main(argv: List[str]):
    word_file = ""
    grid_file = ""

    if "-w" not in argv:
        word_file = "words.txt"
    else:
        if argv.index("-w") < argv.index("-g"):
            word_file = " ".join(argv[argv.index("-w")+1:argv.index("-g")])
        else:
            word_file = " ".join(argv[argv.index("-w")+1:len(argv)])

    print(word_file)
    if "-g" not in argv:
        grid_file = "grid.txt"
    else:
        if argv.index("-w") > argv.index("-g"):
            grid_file = " ".join(argv[argv.index("-g")+1:argv.index("-w")])
        else:
            grid_file = " ".join(argv[argv.index("-g")+1:len(argv)])

    with open(grid_file) as grid_file:
        grid = grid_from_file(grid_file)

    print("Your Grid:")

    print(grid)

    with open(word_file) as words_file:
        words = word_tree_from_file(words_file)

    found_words = find_words(grid, words)

    print(f"{len(found_words)} words found. Longest word: {max(found_words, key=lambda word: len(word.cells))}")

    command = ""
    valid_commands = {
        "semantic search": semantic_search,
        "search containing": search_containing,
        "print longest": get_print_length(True),
        "show path": show_path,
        "print shortest": get_print_length(False),
        "print all": print_all
    }
    while command != "exit":
        command = input("> ")
        if command == "exit":
            break
        words = command.split(" ")
        if len(words) >= 2 and words[0] + " " + words[1] in valid_commands.keys():
            command_func = valid_commands[words[0] + " " + words[1]]
            command_func(words[2:], found_words, grid)
        else:
            print("Unknown command. Should be one of: " + ", ".join(valid_commands.keys())+", exit")


def semantic_search(keys: List[str], found_words: Set[Word], grid: Grid):
    if len(keys) != 2:
        print(f"Expected 2 arguments. Got {len(keys)}.")
        return

    length = keys[1]
    if not length.isdigit():
        print(f"please give an integer count; {length} is not valid")
        return

    transformers.logging.set_verbosity_error()
    model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def get_embedding(word: str) -> torch.Tensor:
        inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)

        return outputs.last_hidden_state.mean(dim=1).squeeze()

    keyword_emb = get_embedding(keys[0])
    word_distances = {}
    for found_word in {str(found_word) for found_word in found_words}:
        word_distances[found_word] = torch.dot(get_embedding(found_word), keyword_emb)
        if len(word_distances) % 100 == 0:
            print(f"processed {len(word_distances)}...")
    print(*[str(item[0]) for item in sorted(word_distances.items(), key=lambda items: items[1], reverse=True)[:int(length)]], sep="\n")


def print_all(keys: List[str], found_words: Set[Word], grid: Grid):
    if len(keys) != 0:
        print(f"Expected 0 arguments. Got {len(keys)}.")
        return
    print(*{str(word) for word in found_words}, sep="\n")


def search_containing(keys: List[str], found_words: Set[Word], grid: Grid):
    if len(keys) != 1:
        print(f"Expected 1 argument. Got {len(keys)}.")
        return
    print(f"Words containing {keys[0]}"+":")
    print(*{str(word) for word in found_words if keys[0].upper() in str(word)}, sep="\n")


def get_print_length(longest: bool) -> Callable[[List[str], Set[Word], Grid], None]:
    def print_length(keys: List[str], found_words: Set[Word], grid: Grid):
        if len(keys) != 1:
            print(f"Expected 1 argument. Got {len(keys)}.")
            return
        length = keys[0]
        if length.isdigit():
            word_list = list({str(word) for word in found_words})
            prompts = sorted(
                word_list,
                key=lambda word: len(word),
                reverse=longest
            )[:int(length)]
            print(*[str(prompt) for prompt in prompts], sep="\n")
        else:
            print(f"please give an integer count; {length} is not valid")

    return print_length


def show_path(keys: List[str], found_words: Set[Word], grid: Grid):
    if len(keys) != 1:
        print(f"Expected 1 argument. Got {len(keys)}.")
        return
    word = keys[0]
    matching_words = {found_word for found_word in found_words if str(found_word) == word.upper()}
    if len(matching_words) > 0:
        for n, matching_word in enumerate(matching_words):
            print(f"Instance {n + 1}")
            print(grid.str_highlighted(matching_word))
    else:
        print(f"There is no word '{word}'")


if __name__ == "__main__":
    main(sys.argv)
