Document Data
=============

Pre-requisite
-------------
Processing document data depends on the optical character recognition (OCR) package: ``tesseract``.


For Ubuntu users, you can install Tesseract and its developer tools by simply running:
    ``sudo apt install tesseract-ocr``

For macOS users, run:
    ``sudo port install tesseract``
or:
    ``brew install tesseract``

For Windows users, installer is available from Tesseract at UB-Mannheim_. 
To access tesseract-OCR from any location you may have to add the directory where the tesseract-OCR binaries are located to the Path variables, probably *C:\Program Files\Tesseract-OCR*.

For additional support, please refer to official instructions for tesseract_

.. _UB-Mannheim: https://github.com/UB-Mannheim/tesseract/wiki
.. _tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html




Quick Start
------------------
.. container:: cards

   .. card::
      :title: AutoMM for Scanned Document Classification - Quick Start
      :link: document_classification.html

      How to use MultiModalPredictor to build a scanned document classifier.


.. toctree::
   :maxdepth: 1
   :hidden:

   document_classification
