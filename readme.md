# Strands Solver

The [New York Times' *Strands*](https://www.nytimes.com/games/strands) puzzle is their most recent addition, but it's already become extremely popular. The game is similar to a word search, but with crossword-like themes connecting the words in each puzzle. 

This tool aims to help users analyze *Strands* puzzles and assist in solving them.

# Getting Started

This tool uses HuggingFace Transformers and PyTorch to support semantic search. Pip install them before running this program.

Run solver.py from the command line like so:

``solver.py [-w lexicon-file] [-g grid-file]``

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`-w lexicon-file` allows you to set a custom lexicon for the solver tool. Simply give it the path to a file containing a list of valid words separated by newlines. If omitted, it will use the defualt words.txt from [UMich](https://websites.umich.edu/~jlawler/wordlist.html)
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`-g grid-file` specifies the grid for this puzzle. Simply give it the path to a txt file containing an ASCII representation of the grid, like so:

```
SCEEBT
KHRSES
NANIGA
HGNRER
TIGOSD
SAISFF
WLNCRY
AYSEEL
```

# Commands

Once you run solver.py, you'll be shown a copy of the grid and a topline report, containing the number of different words found and the longest word among them. You'll then be dropped into a command prompt where you'll have the following options:


## ``semantic search [keyword] [n]``

Uses MiniLM-L6 embeddings to find words relevant to a prompt in the puzzle. Prints the `n`-most relevant words to the `keyword`, in descending order of relevance.

## ``search containing [keyword]``

Returns a list of all valid words in the puzzle that contain the substring `keyword`.

## ``show path [word]``

For a `word` in the puzzle, prints a copy of the grid with the cells needed to assemble the word highlighted. If the word can be achieved multiple ways, it prints all options.

## ``print longest [n]``

Prints the list of words found in the puzzle in descending order by length, up to the `n`-th word.

## ``print shortest [n]``

Prints the list of words found in the puzzle in ascending order by length, up to the `n`-th word.

## ``print all``

Prints all of the words found in the puzzle.

## `exit`

Exits the program.

