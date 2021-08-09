#!/bin/bash
make clean
make
#./test.x
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./report-h100-8000decomp"}' ./test.x

