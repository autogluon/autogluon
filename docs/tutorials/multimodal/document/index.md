# Document Data

## Pre-requisite

Processing document data depends on the optical character recognition (OCR) package `tesseract`.

For Ubuntu users, you can install Tesseract and its developer tools by simply running:

```bash
sudo apt install tesseract-ocr
```

For macOS users, run:

```bash
sudo port install tesseract
```

or run:

```bash
brew install tesseract
```

For Windows users, installer is available from Tesseract at [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). 
To access tesseract-OCR from any location you may have to add the directory where the tesseract-OCR binaries are located to the Path variables.

For additional support, please refer to official instructions for [tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)


## Quick Start

::::{grid} 2
  :gutter: 3

:::{grid-item-card} AutoMM for Scanned Document Classification
  :link: document_classification.html

  How to use MultiModalPredictor to build a scanned document classifier.
:::
::::

```{toctree}
---
caption: Document Data
maxdepth: 1
hidden: true
---

document_classification
```
