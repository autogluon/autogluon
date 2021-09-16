#!/usr/bin/env bash

for arg in "$@"
do
    if [ "$arg" = "shopee" ]; then
        echo kaggle dataset: shopee-iet-machine-learning-competition
        echo data-dir: ./data/shopee-iet-machine-learning-competition
        mkdir -p ./data
        mkdir -p ./data/shopee-iet-machine-learning-competition
        mkdir -p ./data/shopee-iet-machine-learning-competition/train
        cd ./data/shopee-iet-machine-learning-competition/
        kaggle competitions download -c shopee-iet-machine-learning-competition
        unzip shopee-iet-machine-learning-competition.zip
        mv Training\ Images.zip train && cd train && unzip Training\ Images.zip
        cd .. && unzip Test\ Images.zip
        mv Test test
        rm *.zip train/*.zip
        echo shopee-iet-machine-learning-competition dataset finish.

    elif [ "$arg" = "dog" ] ;then
        echo kaggle dataset: dog-breed-identification
        echo data-dir: ./data/dog-breed-identification
        mkdir -p ./data
        mkdir -p ./data/dog-breed-identification
        cd ./data/dog-breed-identification/
        kaggle competitions download -c dog-breed-identification
        unzip dog-breed-identification.zip
        mv train images_all && rm dog-breed-identification.zip
        echo dog-breed-identification dataset finish.

    elif [ "$arg" = "cats" ] ;then
        echo kaggle dataset: dogs-vs-cats-redux-kernels-edition
        echo data-dir: ./data/dogs-vs-cats-redux-kernels-edition
        mkdir -p ./data
        mkdir -p ./data/dogs-vs-cats-redux-kernels-edition
        mkdir -p ./data/dogs-vs-cats-redux-kernels-edition/train
        cd ./data/dogs-vs-cats-redux-kernels-edition/
        kaggle competitions download -c dogs-vs-cats-redux-kernels-edition
        unzip dogs-vs-cats-redux-kernels-edition.zip
        unzip train.zip && unzip test.zip && rm *.zip && mv train images
        cd images && mkdir cat && mkdir dog
        mv cat*jpg cat/ && mv dog*jpg dog/
        echo dogs-vs-cats-redux-kernels-edition dataset finish.

    elif [ "$arg" = "aerial" ] ;then
        echo kaggle dataset: aerial-cactus-identification
        echo data-dir: ./data/aerial-cactus-identification
        mkdir -p ./data
        mkdir -p ./data/aerial-cactus-identification
        cd ./data/aerial-cactus-identification/
        kaggle competitions download -c aerial-cactus-identification
        unzip aerial-cactus-identification.zip
        unzip train.zip && unzip test.zip && mv train images_all
        rm *.zip
        echo aerial-cactus-identification dataset finish.


    elif [ "$arg" = "plant" ] ;then
        echo kaggle dataset: plant-seedlings-classification
        echo data-dir: ./data/plant-seedlings-classification
        mkdir -p ./data
        mkdir -p ./data/plant-seedlings-classification
        cd ./data/plant-seedlings-classification/
        kaggle competitions download -c plant-seedlings-classification
        unzip plant-seedlings-classification.zip
        rm *.zip
        echo plant-seedlings-classification dataset finish.

    elif [ "$arg" = "fish" ] ;then
        echo kaggle dataset: download by yourself

    else
        echo kaggle dataset: download by yourself
    fi


done

