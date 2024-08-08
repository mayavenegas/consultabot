#!/usr/bin/bash

sudo apt -y update
sudo apt -y upgrade
sudo apt -y install net-tools git-lfs
# install grype CVE scanner
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sudo sh -s -- -b /usr/local/bin
# install syft SBOM generator
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sudo sh -s -- -b /usr/local/bin
