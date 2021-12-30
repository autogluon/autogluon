.. note::

    If you don't have them, please first install:
    `XCode <https://developer.apple.com/xcode/>`_, `Homebrew <https://brew.sh>`_, `LibOMP <https://formulae.brew.sh/formula/libomp>`_.
    Once you have Homebrew, LibOMP can be installed via:

    .. code-block:: bash

        # brew install wget
        wget https://raw.githubusercontent.com/Homebrew/homebrew-core/fb8323f2b170bd4ae97e1bac9bf3e2983af3fdb0/Formula/libomp.rb
        brew uninstall libomp
        brew install libomp.rb
        rm libomp.rb

    WARNING: Do not install LibOMP via "brew install libomp" as LibOMP 12 and 13 can cause segmentation faults with LightGBM and XGBoost.
