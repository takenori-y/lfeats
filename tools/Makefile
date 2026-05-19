# Copyright (c) 2026 Takenori Yoshimura
# Released under the MIT License.

PINACT_VERSION  := 3.10.1
TAPLO_VERSION   := 0.10.0
YAMLFMT_VERSION := 0.21.0

all: pinact taplo yamlfmt

pinact:
	mkdir -p pinact
	wget https://github.com/suzuki-shunsuke/pinact/releases/download/v$(PINACT_VERSION)/pinact_linux_amd64.tar.gz -O pinact.tar.gz
	tar xzf pinact.tar.gz -C pinact
	rm -f pinact.tar.gz

taplo:
	mkdir -p taplo
	wget https://github.com/tamasfe/taplo/releases/download/$(TAPLO_VERSION)/taplo-linux-x86_64.gz -O taplo.gz
	gunzip -c taplo.gz > taplo/taplo
	chmod +x taplo/taplo
	rm -f taplo.gz

yamlfmt:
	mkdir -p yamlfmt
	wget https://github.com/google/yamlfmt/releases/download/v$(YAMLFMT_VERSION)/yamlfmt_$(YAMLFMT_VERSION)_Linux_x86_64.tar.gz -O yamlfmt.tar.gz
	tar xzf yamlfmt.tar.gz -C yamlfmt
	rm -f yamlfmt.tar.gz

clean:
	rm -rf pinact taplo yamlfmt

.PHONY: all clean
