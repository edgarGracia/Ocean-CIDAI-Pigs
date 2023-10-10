# Ocean CIDAI Pigs

This repository contains the needed code to publish a pig detector in an Ocean Protocol marketplace.

## Instructions to build the Docker image:
1. Clone the repo:
    
    ```
    git clone https://github.com/edgarGracia/Ocean-CIDAI-Pigs.git
    cd Ocean-CIDAI-Pigs
    ```
3. Download the model and configuration from the repo [releases](https://github.com/edgarGracia/Ocean-CIDAI-Pigs/releases) and place it on ```model/```. E.g.

    ```
    wget https://github.com/edgarGracia/Ocean-CIDAI-Pigs/releases/download/pigs_0_all_det2/Base-RCNN-FPN.yaml -P model/
    wget https://github.com/edgarGracia/Ocean-CIDAI-Pigs/releases/download/pigs_0_all_det2/config.yaml -P model/
    wget https://github.com/edgarGracia/Ocean-CIDAI-Pigs/releases/download/pigs_0_all_det2/model.pth -P model/
    ```

4. Build the Docker image:

    ```
    docker build -t <docker_user>/<repo_name>:<image_tag> .
    ```

## Test the algorithm locally
1. Create the test directories:

    ```
    mkdir -p test/data/inputs
    mkdir -p test/data/outputs
    mkdir -p test/data/inputs/1234
    ```

2. Copy the test data. It must be a zip file with images inside:

    ```cp /path/to/images.zip test/data/inputs/1234/0```

3. Run the previously created docker image or one from the [Docker Hub](https://hub.docker.com/repository/docker/egracia/ocean_cidai_pigs/):
    ```
    docker run --rm \
        -v "$(pwd)/test/data/inputs":/data/inputs \
        -v "$(pwd)/test/data/outputs":/data/outputs \
        -v "$(pwd)/entry_point.sh":/data/transformations/algorithm \
        -e DIDS='["1234"]' \
        --gpus all \
        <docker_image> bash /data/transformations/algorithm
    ```

4. The output data should be saved in ```data/outputs/output.zip```

## Publish the algorithm on an Ocean Protocol marketplace (E.g. [Pontus-X](https://portal.pontus-x.eu/))
1. Push the Docker image to [Docker Hub](https://hub.docker.com/):

    ```docker push <docker_user>/<repo_name>:<image_tag>```

2. Go to the marketplace and publish an asset:
    - Under "Metadata":
        - Set the "Asset Type" to "Algorithm"
        - In the "Docker Image" field select "Custom" and set the name of the previously pushed image
        - In the "Docker Image Entrypoint" field set "bash $ALGO"
    - Under "Access":
        - Select "URL" for the "File" field and paste the URL to the raw Github code of the entry point script. (E.g. https://raw.githubusercontent.com/edgarGracia/cidai_pigs/main/entry_point.sh)


---

The published Docker images can be found at: [https://hub.docker.com/repository/docker/egracia/ocean_cidai_pigs/](https://hub.docker.com/repository/docker/egracia/ocean_cidai_pigs/)
