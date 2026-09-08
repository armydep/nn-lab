# Sample data

## Two corpora, two purposes

| File | Size | Use in |
|---|---|---|
| `tiny.txt` | 144 bytes, 4 lines | Chapters 6–8 |
| `shakespeare.txt` | ~1.1 MB (downloaded) | Chapters 9–12 |

## Why `tiny.txt` is so small

In Chapters 6–7 you are checking whether your bigram counts and your tokenizer are correct. With
four lines you can count occurrences by hand and confirm the model learned what the data
actually says. That verification is the point of the chapter, and a large corpus would make it
impossible.

## Why it is not enough for Chapter 10

`tiny.txt` has ~144 characters and roughly 25 unique ones. A Transformer with even a few
thousand parameters will memorise it perfectly within a few dozen steps — training loss goes to
near zero and generation reproduces the input verbatim.

This is the trap: **it will look like it is working.** Loss drops, text appears. But you will
have built a very expensive lookup table, learned nothing about generalisation, and have no way
to tell a real bug from a data-starvation artefact.

Get a real corpus before Chapter 10:

```bash
python3 sample-data/download_corpus.py
```

## Rough guidance on corpus size

For a character-level model:

- **< 10 KB** — memorises; useful only for verifying the code path runs
- **~100 KB** — begins to produce word-like structure
- **~1 MB** — the practical minimum for output with sentence-like structure; tiny-shakespeare
  sits here
- **> 10 MB** — better results, but slower iteration on a laptop CPU

Any plain-text file of roughly 1 MB works. Project Gutenberg is a good public-domain source if
you would rather train on something other than Shakespeare.

## Watching for the memorisation trap

From Chapter 10 on, hold out ~10% of the text as a validation split and track both losses. When
training loss keeps falling while validation loss rises, the model has started memorising
rather than learning. That gap is the single most informative number in the whole training run,
and it is what Chapter 11 asks you to experiment with.
