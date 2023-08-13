# DPDS_Testing


This repository contains all code created and results obtained in the process of testing the capabilities of the DPDS provenance tool. The code uses and builds upon the original code available at: https://github.com/Lucass97/data_provenance_for_data_science


Requirements:

- All testing conducted in Python 3.11
- Required packages can be found in file: [requirements.txt](requirements.txt)


Running the tests:

- Recommended to run Neo4j through a Docker container composed using the .yml file in neo4j folder: [docker-compose.yml](neo4j/docker-compose.yml)
- Access Neo4j web browser at: http://localhost:7474/browser/
- Ensure working directory is set to testing folder and run code.
- To change between original and updated code change import location of tracker in code. Further details in: [additional_notes.txt](results/additional_notes.txt)


Contents:

- [data](data): Datasets used for testing.
- [misc](misc): Miscellaneous functions used by the provenance tracker.
- [neo4j](neo4j): The .yml file used to compose the Neo4j image in a docker container. Edit to change settings of the Neo4j database. 
- [prov_acquisition](prov_acquisition): The main files used to track provenance and upload information to Neo4j database. Both the [original](prov_acquisition/prov_libraries/provenance_tracker_original.py) and [updated](prov_acquisition/prov_libraries/provenance_tracker.py) tracker files are included, with an additional [pdf file](prov_acquisition/prov_libraries/provenance_tracker_diff.pdf) included to highlight the changes made. 
- [results](results): Contains the results of pipeline survey and all testing conducted. Additional notes are provided for context on selected tests. 
- [testing](testing): All files used to test the DPDS tool. Separate files are included for each of the three datasets.
