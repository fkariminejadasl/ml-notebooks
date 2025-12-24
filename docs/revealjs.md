# Slides from Markdown

This is a tiny workflow to turn a `slides.md` file into a Reveal.js presentation (`slides.html`) using Python.

---

## Files you need

Create two files in the same folder:

- `make_slides.py` (build script)
- `slides.md` (your slide content)

---

## 1) Create `make_slides.py`

Save this as `make_slides.py`:

```python
import pypandoc

# Download pandoc if it isn't available (downloads a local copy pypandoc can use)
pypandoc.download_pandoc()

pypandoc.convert_file(
    "slides.md",
    "revealjs",
    outputfile="slides.html",
    extra_args=[
        "--standalone",
        "-t", "revealjs",
        "-V", "revealjs-url=https://unpkg.com/reveal.js@5",
        "-V", "theme=black",
        "--mathjax",
        "--syntax-highlighting=pygments",
    ],
)

print("Slides written to slides.html")
```

What this does:

* Reads `slides.md`
* Produces a standalone `slides.html`
* Loads Reveal.js from the CDN
* Enables MathJax for formulas
* Enables syntax highlighting for code blocks

---

## 2) Create `slides.md`

Save this as `slides.md`:

````markdown
# Title slide

Your name  
Date

---

## Second slide

- Bullet 1
- Bullet 2

---

## Third slide

You can write **normal markdown** here, formula $\omega$ or code.

$$x^2 + y^2 = 1$$

```python
def f(x):
    return x * x
```
````

---

## 3) Install dependencies

Install pypandoc:

```bash
pip install pypandoc
```

You do not need to install Pandoc separately, because the script downloads it automatically.

---

## 4) Build the slides

Run:

```bash
python make_slides.py
```

This writes `slides.html`.

---

## 5) View the presentation

Open `slides.html` in your browser (double-click it or use your file manager).

Navigation tips:

* Arrow keys: next/previous slide
* `Esc`: overview
* `F`: fullscreen
* URL hash updates are enabled, so slide position is reflected in the URL


