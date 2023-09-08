# CIDAI Pigs

## Instructions to build the Docker image:
1. Clone the repo:
    
    ```git clone ```
2. Download the model and configuration from the repo [releases]() and place it on ```model/```. E.g.

    ```
    wget https://github.com/edgarGracia/cidai_pigs/releases/download/pigs_0_all_det2/Base-RCNN-FPN.yaml -O model/
    wget https://github.com/edgarGracia/cidai_pigs/releases/download/pigs_0_all_det2/config.yaml -O model/
    wget https://github.com/edgarGracia/cidai_pigs/releases/download/pigs_0_all_det2/model.pth -O model/
    ```

3. Build the Docker image:

    ```docker build -t <docker_user>/<repo_name>:<image_tag> .```

## Test the algorithm locally
1. Create the test directories and data:

    ```mkdir ```

## Publish the algorithm on an Ocean Protocol marketplace (E.g. [Pontus-X](https://portal.pontus-x.eu/))
1. Push the Docker image to [Docker Hub](https://hub.docker.com/):

    ```docker push <docker_user>/<repo_name>:<image_tag>```

2. Go to the marketplace and publish an asset:
    - Under "Metadata":
        - Set the "Asset Type" to "Algorithm"
        - In the "Docker Image" field set the name of the previously pushed image
    - Under "Access":
        - Select "URL" for the "File" field and paste the URL to the raw Github code of the entry point script (https://raw.githubusercontent.com/edgarGracia/cidai_pigs/main/entry_point.sh)




Published Docker images: [https://hub.docker.com/repository/docker/egracia/cidai_pigs/general](https://hub.docker.com/repository/docker/egracia/cidai_pigs/general)
